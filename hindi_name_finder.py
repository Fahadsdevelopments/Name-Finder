#!/usr/bin/env python3
"""
Hindi / Marathi Name Finder for voter-list PDFs.

Usage:
    python hindi_name_finder.py /path/to/folder
    python hindi_name_finder.py /path/to/file.pdf
"""

import re, sys, json, hashlib, os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── required packages ────────────────────────────────────────────
try:
    import pdfplumber
except ImportError:
    sys.exit("❌  Run: pip install pdfplumber")

try:
    from indic_transliteration import sanscript
except ImportError:
    sys.exit("❌  Run: pip install indic-transliteration")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import fitz   # PyMuPDF
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False

# ── colours ──────────────────────────────────────────────────────
BOLD="\033[1m"; CYAN="\033[96m"; GREEN="\033[92m"
YELLOW="\033[93m"; RESET="\033[0m"

# ── disk cache dir ───────────────────────────────────────────────
CACHE_DIR = Path.home() / ".hindi_name_finder_cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── OCR thread count (tune to your CPU/RAM) ──────────────────────
OCR_WORKERS = max(1, (os.cpu_count() or 4) // 2)


# ─────────────────────────────────────────────────────────────────
#  STARTUP DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────

def check_dependencies() -> bool:
    ok = True
    if not OCR_AVAILABLE:
        print("❌  pytesseract / Pillow missing. Run: pip install pytesseract Pillow")
        ok = False
    else:
        try:
            ver   = pytesseract.get_tesseract_version()
            langs = pytesseract.get_languages()
            print(f"✅  Tesseract {ver}  —  languages: {', '.join(langs)}")
            if "hin" not in langs:
                print("❌  Hindi pack missing!")
                print("    Re-run the Tesseract installer:")
                print("    https://github.com/UB-Mannheim/tesseract/wiki")
                print("    Tick: Additional language data → hin")
                ok = False
        except Exception as e:
            print(f"❌  Tesseract not found: {e}")
            print("    Install from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("    Add to PATH, e.g.: C:\\Program Files\\Tesseract-OCR")
            ok = False
    if not PYMUPDF_OK:
        print("❌  PyMuPDF missing. Run: pip install pymupdf")
        ok = False
    else:
        print(f"✅  PyMuPDF {fitz.version[0]}")
    return ok


# ─────────────────────────────────────────────────────────────────
#  DISK CACHE  (persists across runs)
# ─────────────────────────────────────────────────────────────────

def _file_hash(pdf_path: Path) -> str:
    """Fast hash: size + mtime, no need to read the whole file."""
    st = pdf_path.stat()
    sig = f"{st.st_size}-{st.st_mtime}"
    return hashlib.md5(sig.encode()).hexdigest()

def _cache_path(pdf_path: Path) -> Path:
    return CACHE_DIR / f"{pdf_path.stem}_{_file_hash(pdf_path)}.json"

def load_disk_cache(pdf_path: Path) -> dict[int, str] | None:
    cp = _cache_path(pdf_path)
    if cp.exists():
        try:
            data = json.loads(cp.read_text(encoding="utf-8"))
            return {int(k): v for k, v in data.items()}
        except Exception:
            pass
    return None

def save_disk_cache(pdf_path: Path, pages: dict[int, str]):
    cp = _cache_path(pdf_path)
    try:
        cp.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────
#  OCR  (parallel pages)
# ─────────────────────────────────────────────────────────────────

def is_garbled(text: str) -> bool:
    if not text or not text.strip():
        return True
    deva = len(re.findall(r'[\u0900-\u097F]', text))
    cids = len(re.findall(r'\(cid:\d+\)', text))
    return cids > 5 or (len(text.strip()) > 50 and deva < 5)

def _ocr_one_page(args: tuple) -> tuple[int, str]:
    """OCR a single page; designed to run in a thread pool."""
    pdf_path, page_idx, total = args
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_idx]
        mat  = fitz.Matrix(150/72, 150/72)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img  = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(
            img, lang="hin+eng", config="--psm 6 --oem 1"
        )
    except Exception as e:
        print(f"\n    ⚠️  Page {page_idx+1} failed: {e}")
        text = ""
    finally:
        doc.close()
    return page_idx + 1, text   # 1-based page number

def ocr_all_pages(pdf_path: Path, n: int) -> dict[int, str]:
    print(f"    ⏳ OCR needed — {n} page(s) using {OCR_WORKERS} thread(s)...", flush=True)
    args = [(pdf_path, i, n) for i in range(n)]
    result: dict[int, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=OCR_WORKERS) as pool:
        futures = {pool.submit(_ocr_one_page, a): a for a in args}
        for fut in as_completed(futures):
            pn, text = fut.result()
            result[pn] = text
            done += 1
            print(f"    ✓ {done}/{n} pages done", end="\r", flush=True)
    print()
    return result


# ─────────────────────────────────────────────────────────────────
#  PAGE LOADING  (memory cache + disk cache)
# ─────────────────────────────────────────────────────────────────

# In-process memory cache: pdf_path str → {page_num: text}
_mem_cache: dict[str, dict[int, str]] = {}

def load_pages(pdf_path: Path) -> dict[int, str]:
    key = str(pdf_path)
    if key in _mem_cache:
        return _mem_cache[key]

    # Try disk cache first
    cached = load_disk_cache(pdf_path)
    if cached is not None:
        _mem_cache[key] = cached
        return cached

    # Read with pdfplumber; decide if OCR is needed
    pages: dict[int, str] = {}
    needs_ocr = False
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            t = page.extract_text() or ""
            if is_garbled(t):
                needs_ocr = True
                break
            pages[i] = t

    if needs_ocr:
        pages = ocr_all_pages(pdf_path, total)

    save_disk_cache(pdf_path, pages)
    _mem_cache[key] = pages
    return pages


# ─────────────────────────────────────────────────────────────────
#  NAME EXTRACTION  (structure-based, no fixed dictionary)
# ─────────────────────────────────────────────────────────────────

_SKIP = {
    'पु','स्त्री','नाव','क्र','नाते','घर','मतदार','संघ','यादी','भाग',
    'क्रमांक','नातेवाईकाचे','पूर्ण','आ','विधानसभा','सेंट्रल','लिंग',
    'वय','मतदाराचे','भाग','और','का','के','की','में','है','हैं','से',
    'को','पर','एक','यह','वह','इस','उस','तथा','एवं','व','ब','प','अ',
    'स','ड','ते','ने','ला','ना','रे','मा','जी','अनु','क्र','सं',
}

DEVA = r'[\u0900-\u097F]+'

# Pre-compiled once at import time
_NAME_RE = re.compile(
    r'(?<!\S)(' + DEVA + r'(?:[ \t]+' + DEVA + r'){1,3})(?!\S)',
    re.UNICODE
)

def extract_names(text: str) -> set[str]:
    found = set()
    for m in _NAME_RE.finditer(text):
        words = m.group().split()
        clean = [w for w in words if w not in _SKIP and len(w) > 1]
        if len(clean) >= 2:
            found.add(' '.join(clean))
    return found

def collect_all_names(pdfs: list) -> tuple[dict, dict[str, dict[int, str]]]:
    """
    Returns:
        locs      — name → [{"file": ..., "page": ...}, ...]
        page_map  — pdf.name → {page_num: text}   (reused for search)
    """
    locs: dict[str, list]          = defaultdict(list)
    page_map: dict[str, dict]      = {}

    for pdf in pdfs:
        if not pdf.exists():
            print(f"  ⚠️  Skipping (not on disk / OneDrive not synced): {pdf.name}")
            continue
        print(f"  📄 {pdf.name}", flush=True)
        try:
            pages = load_pages(pdf)
            page_map[pdf.name] = pages
            for pn, text in pages.items():
                for name in extract_names(text):
                    locs[name].append({"file": pdf.name, "page": pn})
        except Exception as e:
            print(f"  ⚠️  Skipping {pdf.name}: {e}")

    return dict(locs), page_map


# ─────────────────────────────────────────────────────────────────
#  TRANSLITERATION
# ─────────────────────────────────────────────────────────────────

def _word_eng(w: str) -> str:
    r = sanscript.transliterate(w, sanscript.DEVANAGARI, sanscript.ITRANS)
    r = re.sub(r'(?<![AyYvwW])a$', '', r)
    r = r.replace('A','aa').replace('I','ee').replace('U','oo')
    r = r.replace('aa','a').replace('M','n').replace('N','n')
    return r.capitalize()

def eng(name: str) -> str:
    return ' '.join(_word_eng(w) for w in name.split())


# ─────────────────────────────────────────────────────────────────
#  SEARCH  (uses already-loaded page_map — no re-read)
# ─────────────────────────────────────────────────────────────────

def search(pdfs: list, deva_name: str,
           page_map: dict[str, dict[int, str]]) -> list:
    pat  = re.compile(re.escape(deva_name.strip()), re.UNICODE)
    hits = []
    for pdf in pdfs:
        if not pdf.exists():
            continue
        pages = page_map.get(pdf.name)
        if pages is None:
            # Fallback: load from memory/disk cache (no re-OCR)
            try:
                pages = load_pages(pdf)
            except Exception:
                continue
        for pn, text in pages.items():
            for m in pat.finditer(text):
                s   = max(0, m.start()-60)
                e   = min(len(text), m.end()+60)
                ctx = text[s:e].replace('\n',' ').strip()
                hits.append({"file": pdf.name, "page": pn,
                             "matched": m.group(), "context": ctx})
    return hits


# ─────────────────────────────────────────────────────────────────
#  DISPLAY
# ─────────────────────────────────────────────────────────────────

def show_list(locs: dict) -> dict:
    names = sorted(locs, key=lambda n: -len(locs[n]))
    print(f"\n{BOLD}{'═'*58}{RESET}")
    print(f"{BOLD}  {len(names)} unique name(s) found{RESET}")
    print(f"{BOLD}{'═'*58}{RESET}")
    print(f"  {'English':<26} {'Hindi / Marathi':<22} Hits")
    print(f"  {'-'*54}")
    lookup = {}
    for name in names:
        e = eng(name)
        print(f"  {GREEN}{BOLD}{e:<26}{RESET} {name:<22} {len(locs[name])}x")
        lookup[e.lower()] = name
    print(f"{BOLD}{'═'*58}{RESET}")
    return lookup

def show_results(deva: str, hits: list, total: int):
    e = eng(deva)
    print()
    if not hits:
        print(f"{YELLOW}  No matches for '{e}'.{RESET}\n")
        return
    by_file = defaultdict(list)
    for h in hits:
        by_file[h['file']].append(h)
    print(f"{BOLD}{'═'*65}{RESET}")
    print(f"{BOLD}  '{CYAN}{e}{RESET}{BOLD}' ({deva})  —  "
          f"{len(hits)} hit(s) in {len(by_file)}/{total} file(s){RESET}")
    print(f"{BOLD}{'═'*65}{RESET}")
    for fname, fhits in by_file.items():
        print(f"\n  {CYAN}{BOLD}📄 {fname}{RESET}  ({len(fhits)} hit(s))")
        print(f"  {'-'*60}")
        for h in fhits:
            hl = h['context'].replace(
                h['matched'], f"{GREEN}{BOLD}{h['matched']}{RESET}")
            print(f"  Page {h['page']:<5} ↳ ...{hl}...")
    print(f"\n{BOLD}{'═'*65}{RESET}\n")


# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def get_pdfs(target: str) -> list:
    p = Path(target)
    if p.is_file() and p.suffix.lower() == '.pdf':
        return [p]
    if p.is_dir():
        pdfs = sorted(p.glob('*.pdf'))
        if not pdfs:
            sys.exit(f"❌  No PDFs in: {p}")
        return pdfs
    sys.exit(f"❌  Not a file or folder: {target}")

def closest(q: str, lookup: dict) -> list:
    q = q.lower()
    a = [k for k in lookup if k.startswith(q)]
    b = [k for k in lookup if q in k and k not in a]
    return a + b

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    print()
    if not check_dependencies():
        sys.exit("\nFix the issues above and re-run.")
    print()

    pdfs = get_pdfs(sys.argv[1])
    print(f"🔍  Scanning {len(pdfs)} PDF(s)..."
          f"\n    (OCR PDFs take ~5–30s per page on first run)\n")
    print(f"    💾  Disk cache: {CACHE_DIR}\n")

    locs, page_map = collect_all_names(pdfs)
    if not locs:
        print(f"{YELLOW}No names found.{RESET}")
        sys.exit(0)

    lookup = show_list(locs)

    while True:
        print(f"\n  Type a name in English to search "
              f"(or {YELLOW}q{RESET} to quit): ", end='')
        q = input().strip()
        if q.lower() == 'q':
            print("Bye! 👋"); break
        if not q:
            continue
        if q.lower() in lookup:
            show_results(lookup[q.lower()],
                         search(pdfs, lookup[q.lower()], page_map), len(pdfs))
        else:
            matches = closest(q, lookup)
            if not matches:
                print(f"{YELLOW}  '{q}' not found in the list above.{RESET}")
            elif len(matches) == 1:
                deva = lookup[matches[0]]
                print(f"  Closest: {GREEN}{BOLD}{matches[0].title()}{RESET}")
                show_results(deva, search(pdfs, deva, page_map), len(pdfs))
            else:
                print(f"  {YELLOW}Multiple matches — be more specific:{RESET}")
                for m in matches[:10]:
                    print(f"    • {m.title()}")

if __name__ == '__main__':
    main()
