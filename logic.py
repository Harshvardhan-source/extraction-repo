import os
import os.path
import io
import cv2
import csv
import shutil
import pdf2image
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytesseract
import numpy as np
import re
import pandas as pd
from datetime import datetime
import time

# ── Cross-platform Tesseract path ───────────────────────────────────────────
# Uses 'which' / shutil.which to find the system tesseract binary first.
# Falls back to the Windows default path if running on Windows without a
# PATH-accessible binary.  This avoids the previous hardcoded Windows path
# that caused an immediate FileNotFoundError on Linux / Mac.
import shutil as _shutil

_tess_which = _shutil.which("tesseract")
if _tess_which:
    pytesseract.pytesseract.tesseract_cmd = _tess_which
elif os.name == "nt":
    _win_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.isfile(_win_default):
        pytesseract.pytesseract.tesseract_cmd = _win_default
    else:
        # Let pytesseract raise a clear error at OCR time rather than import time
        pytesseract.pytesseract.tesseract_cmd = "tesseract"
else:
    pytesseract.pytesseract.tesseract_cmd = "tesseract"

df = pd.DataFrame(data=None,
                  columns=["id", "page", "split", "polling station", "polling address", "voterid", "name", "father",
                           "address", "age", "gender"])

global origfile
global splitname
global splitnum
global pollingstation
global pollingaddress
global assembly_constituency_no
global assembly_constituency_name
global section_no_and_name
global part_no

assembly_constituency_no   = ""
assembly_constituency_name = ""
section_no_and_name        = ""
part_no                    = ""

# Voter ID pattern: 2-3 uppercase letters then a DIGIT (or OCR 'O') then 5-7 more digits.
# Requiring a digit right after the letters prevents city names (MANGALORE, SURATHKAL …)
# from matching — they have no digits at all.
VOTER_ID_RE     = re.compile(r'\b[A-Z]{2,3}[0-9O]\d{5,7}\b')
VOTER_ID_STRICT = re.compile(r'^[A-Z]{2,3}\d{6,8}$')
_FIELD_KEYWORDS = ['Name', 'House', 'Age', 'Gender', 'Father', 'Husband', 'Mother', 'Wife', 'Other']


def safe_ocr_resize(img, target_scale=2.5, max_long_edge=4500):
    """Resize image for OCR while capping the long edge to avoid Tesseract memory errors."""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    scale = min(target_scale, max_long_edge / long_edge)
    if scale <= 0:
        scale = 1.0
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)


def safe_tesseract_ocr(img, config):
    """Run Tesseract OCR with progressive fallback on memory errors (75%→50%→25%)."""
    attempt_img = img
    for shrink in (1.0, 0.75, 0.5, 0.25):
        if shrink < 1.0:
            hh, ww = img.shape[:2]
            attempt_img = cv2.resize(img,
                                     (max(1, int(ww * shrink)), max(1, int(hh * shrink))),
                                     interpolation=cv2.INTER_AREA)
        try:
            return pytesseract.image_to_string(attempt_img, config=config)
        except Exception as exc:
            if shrink == 0.25:
                print(f"  [OCR] All retries failed: {exc}")
                return ""
            print(f"  [OCR] Memory error at {shrink:.0%} size, retrying smaller…")
    return ""


def fix_voter_id(v):
    """Correct common OCR misreads in a raw voter ID string."""
    if not v or (isinstance(v, float) and pd.isna(v)):
        return v
    v = str(v).strip()
    v = re.sub(r'^\$', 'S', v)           # $TN  → STN
    v = re.sub(r'^W2Z', 'WZZ', v)        # W2Z  → WZZ
    v = re.sub(r'^T[™®©]N', 'TUN', v)   # T™N  → TUN
    if len(v) >= 4 and v[:3].isalpha() and v[3] == 'O':
        v = v[:3] + '0' + v[4:]          # STNO → STN0
    if len(v) == 10 and v[:3].isalpha() and v[3:9].isdigit() and v[-1] == 'S':
        v = v[:9] + '5'                  # trailing S → 5

    # ── 2-letter → 3-letter prefix: OCR frequently drops the first letter ────
    if re.match(r'^TN[0-9O]', v):
        v = 'S' + v
    elif re.match(r'^WZ[0-9O]', v):
        v = 'L' + v
    elif re.match(r'^LW\d{8}$', v):
        v = 'LWZ' + v[3:]

    if re.search(r'[a-z\[\]{}\s™©®]', v) or len(v) < 8:
        return None                       # garbage — discard
    return v


def _scan_top_right_voter_id(col_img_path):
    """
    MASTER FIX — Targeted OCR on the TOP-RIGHT corner of a column strip.

    The voter ID for the FIRST card in each column always sits at:
        • top 18 % of the strip height
        • right 45 % of the strip width
    A dedicated scan on the ORIGINAL (unscaled) image with a strict
    character whitelist is far more reliable than the old top-strip
    prepend approach (which caused duplicate entries by re-reading the
    whole first card and feeding it into the token stream twice).

    Returns the voter ID string (already fix_voter_id-cleaned) or None.
    """
    img = cv2.imread(col_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        return None

    # Crop: top 18 % × right 55 %
    y2 = max(1, int(h * 0.18))
    x1 = int(w * 0.45)
    crop = img[0:y2, x1:w]
    if crop.size == 0:
        return None

    # Upscale 4× — critical for reading small alphanumeric text reliably
    crop = cv2.resize(crop, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)

    # Clean binary image via adaptive threshold
    crop = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)

    # Pass 1: single-line mode, strict A-Z 0-9 whitelist
    _TESS_VID = r'--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    try:
        raw = pytesseract.image_to_string(crop, config=_TESS_VID).strip().upper()
        m = VOTER_ID_RE.search(raw)
        if m:
            vid = fix_voter_id(m.group())
            if vid:
                return vid
    except Exception:
        pass

    # Pass 2: uniform-block mode fallback (handles multi-line crops)
    _TESS_VID6 = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    try:
        raw2 = pytesseract.image_to_string(crop, config=_TESS_VID6).strip().upper()
        m2 = VOTER_ID_RE.search(raw2)
        if m2:
            vid2 = fix_voter_id(m2.group())
            if vid2:
                return vid2
    except Exception:
        pass

    return None


def _detect_card_starts(img, n_cards):
    """
    Detect the real y-coordinate where each of n_cards starts in a column
    strip image, by finding the strip's horizontal border lines.
    Falls back to naive uniform division (h // n_cards) if detection fails.
    Returns: list of n_cards y-coordinates (ints), card_starts[0] == 0.
    """
    h, w = img.shape[:2]
    fallback = [int(i * h / n_cards) for i in range(n_cards)]

    try:
        row_darkness = (img < 128).sum(axis=1)
        threshold = w * 0.5
        border_rows = [i for i in range(h) if row_darkness[i] > threshold]
        if len(border_rows) < 2:
            return fallback

        clusters = []
        cur = [border_rows[0]]
        for r in border_rows[1:]:
            if r - cur[-1] <= 5:
                cur.append(r)
            else:
                clusters.append(int(np.mean(cur)))
                cur = [r]
        clusters.append(int(np.mean(cur)))

        if len(clusters) < 2:
            return fallback

        collapsed = []
        i = 0
        while i < len(clusters):
            if i + 1 < len(clusters) and (clusters[i + 1] - clusters[i]) <= 30:
                collapsed.append(clusters[i + 1])
                i += 2
            else:
                collapsed.append(clusters[i])
                i += 1

        if len(collapsed) < 2:
            return fallback

        pitches = [collapsed[j + 1] - collapsed[j] for j in range(len(collapsed) - 1)]
        pitches = [p for p in pitches if p > 60]
        if len(pitches) < 1:
            return fallback
        pitch = float(np.median(pitches))

        anchor = collapsed[0]
        starts = [0]
        y = float(anchor)
        for _ in range(n_cards - 1):
            starts.append(int(round(y)))
            y += pitch

        if len(starts) != n_cards or any(
                starts[i] >= starts[i + 1] for i in range(len(starts) - 1)):
            return fallback
        if starts[-1] >= h:
            return fallback

        return starts
    except Exception:
        return fallback


def _scan_all_ages_in_strip(col_img_path, n_cards=10):
    """
    SPEED OPTIMISATION - Extract age/gender for every card in a column strip
    with a SINGLE Tesseract call instead of n_cards separate calls.
    Returns: [(age_str, gender_str), ...] - one tuple per card, None where not found.
    """
    img = cv2.imread(col_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [(None, None)] * n_cards
    h, w = img.shape[:2]
    if h < 50:
        return [(None, None)] * n_cards

    card_starts = _detect_card_starts(img, n_cards)

    SCALE = 3.0
    OFFSET_TOP, OFFSET_BOT = 100, 150

    strips = []
    cum_h  = [0]

    for ci in range(n_cards):
        s    = card_starts[ci]
        y1   = min(h, s + OFFSET_TOP)
        y2   = min(h, s + OFFSET_BOT)
        crop = img[y1:y2, 0:max(1, int(w * 0.85))]
        if crop.size == 0:
            # FIX: placeholder MUST use the same width as real crops so that
            # np.vstack doesn't crash with a dimension-1 mismatch.
            # Old code used (10, 100) — after SCALE=3× resize → width 300.
            # Real crops are int(w*0.85) wide → after 3× resize → width ~1485.
            # Making the placeholder the same width (int(w*0.85)) fixes it.
            crop = np.ones((10, max(1, int(w * 0.85))), dtype=np.uint8) * 255
        crop = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        crop = cv2.adaptiveThreshold(crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 5)
        strips.append(crop)
        cum_h.append(cum_h[-1] + crop.shape[0])

    combined = np.vstack(strips)

    try:
        data = pytesseract.image_to_data(
            combined, config='--psm 6',
            output_type=pytesseract.Output.DICT)
    except Exception:
        return [(None, None)] * n_cards

    card_words = [[] for _ in range(n_cards)]
    for i, word in enumerate(data['text']):
        word = str(word).strip()
        if not word or int(data['conf'][i]) < 0:
            continue
        top = int(data['top'][i])
        for ci in range(n_cards):
            if cum_h[ci] <= top < cum_h[ci + 1]:
                card_words[ci].append(word)
                break

    _AGE  = r'(?:Age|Ago|Aqe|A[gq9G][e3Eo]?)'
    _GEND = r'(?:Gender|Gend[e3]?r?|G[e3]nd[e3]?r?)'
    results = []
    for ci in range(n_cards):
        text = ' '.join(card_words[ci])
        age, gender = None, None
        m = re.search(
            _AGE + r'\s*[^A-Za-z0-9]{0,4}(\d{1,3})[^A-Za-z0-9]{0,12}'
            + _GEND + r'\s*[^A-Za-z0-9]{0,4}([A-Za-z]+)',
            text, re.IGNORECASE)
        if m:
            av = int(m.group(1));  age = str(av) if 1 <= av <= 110 else None
            gender = m.group(2).capitalize()
        else:
            ma = re.search(_AGE + r'\s*[^A-Za-z0-9]{0,4}(\d{1,3})', text, re.IGNORECASE)
            if ma:
                av = int(ma.group(1));  age = str(av) if 1 <= av <= 110 else None
            mg = re.search(_GEND + r'\s*[^A-Za-z0-9]{0,4}([A-Za-z]+)', text, re.IGNORECASE)
            if mg: gender = mg.group(1).capitalize()
            if not ma:
                mn = re.search(
                    r'(?<![/\-#\d])(\d{1,3})\s{0,6}(?:Male|Female)', text, re.IGNORECASE)
                if mn:
                    av = int(mn.group(1));  age = str(av) if 1 <= av <= 110 else None
            if not gender:
                mg2 = re.search(r'\b(Male|Female)\b', text, re.IGNORECASE)
                if mg2: gender = mg2.group(1).capitalize()
        results.append((age, gender))
    return results


def pdfs_identification(input_dir):
    listoffiles = []
    extDict = (".jpeg", ".jpg", ".png")

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".pdf".lower()):
                listoffiles.append(filename)
            elif filename.lower().endswith(tuple(extDict)):
                os.remove(os.path.join(root, filename))

    return listoffiles


def extract_polling_station_details(file):
    global pollingstation
    global pollingaddress
    global assembly_constituency_no
    global assembly_constituency_name
    global section_no_and_name
    global part_no

    img  = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # ── Pass 1: tiny top-of-page crop ───────────────────────────────────────
    top_region = gray[:max(1, int(h * 0.30)), :]
    top_small  = safe_ocr_resize(top_region, target_scale=1.0, max_long_edge=900)
    top_text   = safe_tesseract_ocr(top_small, config='--psm 4').replace('\n\n', '\n')
    top_lines  = top_text.split('\n')

    for line in top_lines:
        if 'Assembly Constituency' in line:
            m = re.search(r':\s*(\d+)[- ]+(.+?)(?:\s{2,}|Part|$)', line)
            if m:
                assembly_constituency_no   = m.group(1).strip()
                assembly_constituency_name = m.group(2).strip()
            break

    _PART_RE = re.compile(r'\bPart\s+No[^A-Za-z0-9]{0,5}(\d+)', re.IGNORECASE)
    for line in top_lines:
        _pm = _PART_RE.search(line)
        if _pm:
            part_no = _pm.group(1).strip()
            break

    for line in top_lines:
        if 'Section No and Name' in line:
            m = re.search(r'Section No and Name\s*[:\-]?\s*(.+)', line, re.IGNORECASE)
            if m:
                section_no_and_name = m.group(1).strip()
            break

    # ── Pass 2: full page at low resolution ─────────────────────────────────
    full_small = safe_ocr_resize(gray, target_scale=1.0, max_long_edge=1400)
    full_text  = safe_tesseract_ocr(full_small, config='--psm 4').replace('\n\n', '\n')
    all_lines  = full_text.split('\n')

    if not assembly_constituency_no:
        for line in all_lines:
            if 'Assembly Constituency' in line:
                m = re.search(r':\s*(\d+)[- ]+(.+?)(?:\s{2,}|Part|$)', line)
                if m:
                    assembly_constituency_no   = m.group(1).strip()
                    assembly_constituency_name = m.group(2).strip()
                break

    if not part_no:
        for line in all_lines:
            _pm2 = _PART_RE.search(line)
            if _pm2:
                part_no = _pm2.group(1).strip()
                break

    # Polling station name — row after "Male/Female" column header
    try:
        firstrow = next(
            i for i, v in enumerate(all_lines)
            if 'male/female' in v.lower()
        ) + 1
        pollingstation = all_lines[firstrow] if firstrow < len(all_lines) else ''
    except StopIteration:
        pollingstation = ''

    # Polling address — row after "Stations in this part" column header
    try:
        secondrow = next(
            i for i, v in enumerate(all_lines)
            if 'stations in this part' in v.lower()
        ) + 1
        pollingaddress = all_lines[secondrow] if secondrow < len(all_lines) else ''
    except StopIteration:
        pollingaddress = ''


def imgcrop(input_file, indicator):
    """
    Crop a page image into 3 column strips and save them under a 'Splits'
    subdirectory next to the source image.

    Cross-platform: uses os.path.join / os.makedirs instead of Windows-only
    backslash concatenation.
    """
    page = []

    filename, file_extension = os.path.splitext(os.path.basename(input_file))
    # FIX: was using backslash concatenation (Windows-only).
    #      os.path.join works on all platforms.
    splits_dir = os.path.join(os.path.dirname(input_file), 'Splits')
    os.makedirs(splits_dir, exist_ok=True)

    if indicator == 1:
        reqcut = 90
    else:
        reqcut = 200

    im = Image.open(input_file)
    box1 = (0, reqcut, 570, 2230)
    a = im.crop(box1)
    out0 = os.path.join(splits_dir, f"{filename}-0-0{file_extension}")
    a.save(out0)
    page.append(out0)

    box2 = (571, reqcut, 1070, 2230)
    b = im.crop(box2)
    out1 = os.path.join(splits_dir, f"{filename}-1-1{file_extension}")
    b.save(out1)
    page.append(out1)

    box3 = (1070, reqcut, 1653, 2230)
    c = im.crop(box3)
    out2 = os.path.join(splits_dir, f"{filename}-2-2{file_extension}")
    c.save(out2)
    page.append(out2)

    return page


def pageSplit(input_dir, filename, poppler_path):
    global origfile
    global splitname
    global splitnum
    global pollingstation
    global pollingaddress
    global assembly_constituency_no
    global assembly_constituency_name
    global section_no_and_name
    global part_no

    origfile = filename
    pagelist = []

    input_page_path = os.path.join(input_dir, 'pages')
    os.makedirs(input_page_path, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 1 — PDF → JPEG images
    # PyMuPDF (fitz) is 3-5x faster than pdf2image/poppler because it runs
    # in-process (no subprocess).  Falls back to the original pdf2image path
    # if PyMuPDF is not installed.  Install once:  pip install PyMuPDF
    # ════════════════════════════════════════════════════════════════════════
    print("Current Process -- PDF to Image Split")
    _pdf_full = os.path.join(input_dir, filename)

    try:
        import fitz as _fitz                         # PyMuPDF fast path
        _doc = _fitz.open(_pdf_full)
        _mat = _fitz.Matrix(200 / 72, 200 / 72)     # 200 DPI
        for _i in range(len(_doc)):
            _pix = _doc[_i].get_pixmap(matrix=_mat, colorspace=_fitz.csRGB)
            _out = os.path.join(input_page_path, f'page{_i}.jpg')
            _pix.save(_out)
            pagelist.append(_out)
        _doc.close()
        print(f"  PyMuPDF: {len(pagelist)} pages rendered")
    except ImportError:
        # Fallback: original poppler/pdf2image (slower, no extra install needed)
        _imgs = convert_from_path(_pdf_full, fmt="jpeg", dpi=200,
                                  jpegopt={'quality': 100}, poppler_path=poppler_path)
        for _i, _img in enumerate(_imgs):
            _out = os.path.join(input_page_path, f'page{_i}.jpg')
            _img.save(_out, 'JPEG')
            pagelist.append(_out)

    extract_polling_station_details(pagelist[0])

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Header detection + column cropping  (sequential, fast)
    # ════════════════════════════════════════════════════════════════════════
    strip_task_list = []

    for pslit in range(len(pagelist)):
        img_p = cv2.imread(pagelist[pslit], cv2.IMREAD_GRAYSCALE)
        h, w  = img_p.shape[:2]

        header_region = img_p[:max(1, int(h * 0.08)), :]
        header_small  = safe_ocr_resize(header_region, target_scale=1.5, max_long_edge=2500)
        inittext      = safe_tesseract_ocr(header_small, config='--psm 6').replace('\n\n', '\n')
        maintext      = inittext.split("\n")
        page_lower    = inittext.lower()

        for hline in maintext[:8]:
            if "Section No and Name" in hline:
                m = re.search(r'Section No and Name\s*[:\-]?\s*(.+)', hline, re.IGNORECASE)
                if m:
                    section_no_and_name = m.group(1).strip()
                break
        for hline in maintext[:8]:
            _pm = re.search(r'\bPart\s+No[^A-Za-z0-9]{0,5}(\d+)', hline, re.IGNORECASE)
            if _pm:
                part_no = _pm.group(1).strip()
                break

        has_sn  = "section no and name" in page_lower
        has_loa = "list of additions"   in page_lower
        has_ac  = "assembly constituency" in page_lower

        if has_sn:
            strips = imgcrop(pagelist[pslit], 1)
        elif has_loa:
            strips = imgcrop(pagelist[pslit], 2)
        elif has_ac:
            strips = imgcrop(pagelist[pslit], 1)
        else:
            print(f"    ↷  Skipping non-voter page: {os.path.basename(pagelist[pslit])}")
            continue

        _snap = {
            'pollingstation':             pollingstation,
            'pollingaddress':             pollingaddress,
            'assembly_constituency_no':   assembly_constituency_no,
            'assembly_constituency_name': assembly_constituency_name,
            'section_no_and_name':        section_no_and_name,
            'part_no':                    part_no,
        }
        for sp in strips:
            strip_task_list.append((sp, _snap))

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 3 — PARALLEL OCR
    # ════════════════════════════════════════════════════════════════════════
    _N = len(strip_task_list)
    print(f"Current Process -- Data Extraction from Images  ({_N} strips, 3 parallel workers)")

    def _ocr_strip(strip_path):
        """All OCR work for one column strip — self-contained, thread-safe."""
        col_ag    = _scan_all_ages_in_strip(strip_path)
        img_raw   = cv2.imread(strip_path, cv2.IMREAD_GRAYSCALE)
        img_sc    = safe_ocr_resize(img_raw, target_scale=2.5, max_long_edge=3500)
        text      = safe_tesseract_ocr(img_sc, config='--psm 11').replace('\n\n', '\n')

        # Only fire the expensive targeted voter-ID scan when the main text
        # doesn't already contain a valid voter ID near the top of the strip.
        _first_500 = text[:500].upper()
        _has_vid   = bool(VOTER_ID_RE.search(_first_500))
        top_vid    = None if _has_vid else _scan_top_right_voter_id(strip_path)

        return strip_path, col_ag, text, top_vid

    strip_results = {}
    with ThreadPoolExecutor(max_workers=3) as _pool:
        _futs = {_pool.submit(_ocr_strip, sp): sp for sp, _ in strip_task_list}
        for _fut in as_completed(_futs):
            sp, col_ag, text, top_vid = _fut.result()
            strip_results[sp] = (col_ag, text, top_vid)

    # ════════════════════════════════════════════════════════════════════════
    # PHASE 4 — Parse OCR text into rows
    # ════════════════════════════════════════════════════════════════════════
    df1 = pd.DataFrame(data=None,
                       columns=["id", "page", "split", "polling station", "polling address",
                                "voterid", "serial_no", "name", "father",
                                "Relative Name", "Relation Type",
                                "address", "age", "gender",
                                "assembly_constituency_no", "assembly_constituency_name",
                                "section_no_and_name", "part_no"])

    for strip_path, _snap in strip_task_list:
        # Restore per-page globals from the snapshot taken in Phase 2
        pollingstation             = _snap['pollingstation']
        pollingaddress             = _snap['pollingaddress']
        assembly_constituency_no   = _snap['assembly_constituency_no']
        assembly_constituency_name = _snap['assembly_constituency_name']
        section_no_and_name        = _snap['section_no_and_name']
        part_no                    = _snap['part_no']

        col_ag, text, top_vid = strip_results[strip_path]
        df1 = pd.concat([df1, dataextraction(strip_path, text, col_ag, top_vid)], axis=0)

    df1.reset_index(inplace=True, drop=True)

    # Relation Type / Relative Name split
    try:
        def _relation_type(val):
            if pd.isna(val): return None
            v = str(val).lower()
            if "father"  in v: return 'Father'
            if "husband" in v: return 'Husband'
            if "mother"  in v: return 'Mother'
            if "wife"    in v: return 'Wife'
            if "other"   in v: return 'Other'
            if "legal"   in v: return 'Legal'
            return 'Unknown'

        def _relative_name(val):
            """
            FIX: old regex used 'Fathers?' which never matched "Father's"
            (apostrophe broke it), so 100% of values kept the full prefix.
            New approach: just grab everything after 'Name :' / 'Name -'.
            Handles: "Father's Name : X", "Husband's Name : X", "Mothers Name : X", etc.
            """
            if pd.isna(val): return None
            v = str(val).strip()
            # Grab the actual name — everything after the last "Name :" occurrence
            m = re.search(r'\bName\s*[:\-]\s*(.+)$', v, re.IGNORECASE)
            if m:
                return m.group(1).strip()
            return v if v else None

        df1['Relation Type']  = df1['father'].apply(_relation_type)
        df1['Relative Name']  = df1['father'].apply(_relative_name)
    except Exception:
        pass

    return df1


def dataextraction(col_img_path, text, col_age_gender, top_vid=None):
    """Parse raw OCR text from one column strip into a DataFrame of voter records."""
    global origfile
    global splitname
    global splitnum
    global pollingstation
    global pollingaddress
    global assembly_constituency_no
    global assembly_constituency_name
    global section_no_and_name
    global part_no

    splitname = os.path.basename(col_img_path)
    splitnum  = splitname

    df2 = pd.DataFrame(data=None,
                       columns=["id", "page", "split", "polling station", "polling address",
                                "voterid", "serial_no", "name", "father",
                                "Relative Name", "Relation Type",
                                "address", "age", "gender",
                                "assembly_constituency_no", "assembly_constituency_name",
                                "section_no_and_name", "part_no"])

    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]

    # ── Compiled regexes used inside the parsing loop ────────────────────────
    # Matches "Name :", "Voter Name :", OCR variants "Nam :", "Namo :", "Name-"
    _NAME_LINE_RE = re.compile(
        r'^(?:Voter\s*)?Na(?:me?|im?e?)\s*[:\-]\s*', re.IGNORECASE
    )
    # Lines that are clearly NOT a voter name (field labels, IDs, numbers)
    _NOT_A_NAME_RE = re.compile(
        r'^(?:House|Age|Gender|Father|Husband|Mother|Wife|Other|Legal|'
        r'Serial|Sl\.?|S\.?No|Part|Section|Assembly|Polling|Address|\d)',
        re.IGNORECASE
    )

    # ── Extract voter records from OCR text ──────────────────────────────────
    records = []
    current = {}
    # Candidate name lines: unrecognised text after a voter ID, before any
    # labelled field.  Used as a fallback when the "Name :" label is missing.
    _name_candidates: list[str] = []

    def _flush():
        if current:
            # ── Name recovery: if no name was found but we collected
            #    unclassified text lines between voter ID and first field,
            #    pick the best candidate (most alphabetic characters).
            if 'name' not in current and _name_candidates:
                # Filter: 2–50 chars, mostly letters/spaces, not a voter ID
                good = [
                    c for c in _name_candidates
                    if 2 <= len(c) <= 50
                    and sum(ch.isalpha() or ch == ' ' for ch in c) / max(len(c), 1) >= 0.75
                    and not VOTER_ID_RE.search(c.upper())
                ]
                if good:
                    # Prefer longer candidates (more complete names)
                    current['name'] = max(good, key=len)
            records.append(dict(current))
            current.clear()
        _name_candidates.clear()

    for line in lines:
        # ── Voter ID line ─────────────────────────────────────────────────────
        vid_m = VOTER_ID_RE.search(line.upper())
        if vid_m:
            _flush()
            vid = fix_voter_id(vid_m.group())
            current['voterid'] = vid
            # Serial number sometimes appears on the same line before the ID
            sn_m = re.search(r'^\s*(\d{1,4})\b', line)
            if sn_m:
                current['serial_no'] = sn_m.group(1)
            continue

        # ── Serial number only line ───────────────────────────────────────────
        if re.match(r'^\s*\d{1,4}\s*$', line):
            sn = line.strip()
            if current:
                current.setdefault('serial_no', sn)
            else:
                _flush()
                current['serial_no'] = sn
            continue

        # ── Name line — "Name : X" or OCR variants ───────────────────────────
        if _NAME_LINE_RE.match(line):
            val = _NAME_LINE_RE.sub('', line).strip()
            if val:
                current['name'] = val
            continue

        # ── House / address line ──────────────────────────────────────────────
        if re.match(r'^House\s*(?:No\.?|Number|#)?\s*[:\-]', line, re.IGNORECASE):
            val = re.sub(r'^House\s*(?:No\.?|Number|#)?\s*[:\-]\s*', '',
                         line, flags=re.IGNORECASE).strip()
            current['address'] = val
            continue

        # ── Age / Gender line ─────────────────────────────────────────────────
        age_m = re.search(r'\bAge\s*[:\-]?\s*(\d{1,3})', line, re.IGNORECASE)
        gen_m = re.search(r'\bGender\s*[:\-]?\s*(Male|Female)', line, re.IGNORECASE)
        if age_m or gen_m:
            if age_m:
                current['age'] = age_m.group(1)
            if gen_m:
                current['gender'] = gen_m.group(1).capitalize()
            continue

        # ── Relation line — Father/Husband/Mother/Wife/Other ─────────────────
        rel_m = re.match(
            r'^(Father|Husband|Mother|Wife|Other|Legal\s+Guardian)(?:\'?s?\s+Name)?\s*[:\-]\s*(.+)',
            line, re.IGNORECASE)
        if rel_m:
            current['father'] = f"{rel_m.group(1)}'s Name : {rel_m.group(2).strip()}"
            continue

        # ── Unclassified line — collect as potential name candidate ───────────
        # Only collect when we're inside a card (voter ID seen) and haven't
        # found the name yet, and the line doesn't look like a field label.
        if current and 'voterid' in current and 'name' not in current:
            if not _NOT_A_NAME_RE.match(line):
                _name_candidates.append(line)

    _flush()  # flush last record

    # ── Build DataFrame from records ──────────────────────────────────────────
    for idx, rec in enumerate(records):
        age_val    = rec.get('age')
        gender_val = rec.get('gender')

        # Fallback from the combined age/gender strip scan
        if (not age_val or not gender_val) and idx < len(col_age_gender):
            fa, fg = col_age_gender[idx]
            if not age_val and fa:
                age_val = fa
            if not gender_val and fg:
                gender_val = fg

        row = {
            "id":                        origfile,
            "page":                      os.path.basename(col_img_path).split('-')[0],
            "split":                     splitnum,
            "polling station":           pollingstation,
            "polling address":           pollingaddress,
            "voterid":                   rec.get('voterid'),
            "serial_no":                 rec.get('serial_no'),
            "name":                      rec.get('name', 'not found'),
            "father":                    rec.get('father', ''),
            "Relative Name":             None,
            "Relation Type":             None,
            "address":                   rec.get('address', ''),
            "age":                       age_val,
            "gender":                    gender_val,
            "assembly_constituency_no":  assembly_constituency_no,
            "assembly_constituency_name": assembly_constituency_name,
            "section_no_and_name":       section_no_and_name,
            "part_no":                   part_no,
        }
        df2 = pd.concat([df2, pd.DataFrame([row])], axis=0)

    # ── Inject top_vid into the FIRST record if the main scan missed it ──────
    if top_vid and not df2.empty:
        first_idx = df2.index[0]
        if pd.isna(df2.at[first_idx, 'voterid']) or df2.at[first_idx, 'voterid'] is None:
            df2.at[first_idx, 'voterid'] = top_vid

    df2.reset_index(inplace=True, drop=True)
    return df2


def excel_read(filepath):
    hindu, christian, muslim = [], [], []
    try:
        _df = pd.read_excel(filepath, sheet_name=0)
        hindu = [str(x).lower() for x in _df["Names"] if pd.notna(x)]
    except Exception as e:
        print(f"Hindu file is not read properly: {e}")
    try:
        _df = pd.read_excel(filepath, sheet_name=1)
        christian = [str(x).lower() for x in _df["Names"] if pd.notna(x)]
    except Exception as e:
        print(f"Christian file is not read properly: {e}")
    try:
        _df = pd.read_excel(filepath, sheet_name=2)
        muslim = [str(x).lower() for x in _df["Names"] if pd.notna(x)]
    except Exception as e:
        print(f"Muslim file is not read properly: {e}")
    return hindu, christian, muslim


def caste_function(df3, caste_file):
    df = df3.copy(deep=True)
    caste_df = pd.read_excel(caste_file, sheet_name='Caste')
    caste_df.columns = ["Sub_Caste", "caste"]
    new_df = pd.merge(df, caste_df, how='left', left_on='sub_caste', right_on='Sub_Caste')
    return new_df


def religion_update(df2, hindu, christian, muslim, source):
    df = df2.copy(deep=True)
    rowcount = 0
    for name in df["father"]:
        for i in range(len(hindu)):
            if hindu[i].lower() in str(name).lower():
                df.at[rowcount, "religion"] = "hindu"
                df.at[rowcount, "key_identifier"] = str(hindu[i])
                df.at[rowcount, "source"] = source
                break
        rowcount += 1

    rowcount = 0
    for name in df["name"]:
        for i in range(len(hindu)):
            if hindu[i].lower() in str(name).lower():
                df.at[rowcount, "religion"] = "hindu"
                df.at[rowcount, "key_identifier"] = str(hindu[i])
                df.at[rowcount, "source"] = source
                break
        rowcount += 1

    rowcount = 0
    for name in df["father"]:
        for i in range(len(christian)):
            if christian[i].lower() in str(name).lower():
                df.at[rowcount, "religion"] = "christian"
                df.at[rowcount, "key_identifier"] = str(christian[i])
                df.at[rowcount, "source"] = source
                break
        rowcount += 1

    rowcount = 0
    for name in df["name"]:
        for i in range(len(christian)):
            if christian[i].lower() in str(name).lower():
                df.at[rowcount, "religion"] = "christian"
                df.at[rowcount, "key_identifier"] = str(christian[i])
                df.at[rowcount, "source"] = source
                break
        rowcount += 1

    rowcount = 0
    for name in df["father"]:
        for i in range(len(muslim)):
            if muslim[i].lower() in str(name).lower():
                df.at[rowcount, "religion"] = "muslim"
                df.at[rowcount, "key_identifier"] = str(muslim[i])
                df.at[rowcount, "source"] = source
                break
        rowcount += 1

    rowcount = 0
    for name in df["name"]:
        for i in range(len(muslim)):
            if muslim[i].lower() in str(name).lower():
                df.at[rowcount, "religion"] = "muslim"
                df.at[rowcount, "key_identifier"] = str(muslim[i])
                df.at[rowcount, "source"] = source
                break
        rowcount += 1

    return df


def sub_caste_function(df3, caste_file):
    try:
        df = df3.copy(deep=True)
        df["name"] = df["name"].fillna("not available")
        df_name = pd.ExcelFile(caste_file)

        for sheet in df_name.sheet_names:
            if sheet == "Caste":
                continue
            else:
                caste_df = pd.read_excel(caste_file, sheet_name=sheet)

                if len(caste_df) > 0:
                    caste_list = list(caste_df["Names"])
                    caste_list = [str(x).lower() for x in caste_list]

                    rowcount = 0
                    for name in df["father"]:
                        for i in range(len(caste_list)):
                            if name != "":
                                if caste_list[i].lower() in str(name).lower():
                                    df.at[rowcount, "sub_caste"] = sheet
                                    break
                        rowcount += 1

                    rowcount = 0
                    for name in df["name"]:
                        if name and not pd.isnull(name):
                            for i in range(len(caste_list)):
                                if caste_list[i].lower() in str(name).lower():
                                    df.at[rowcount, "sub_caste"] = sheet
                                    break
                        rowcount += 1

    except Exception as e:
        print(f'Error while reading caste file: {e}')

    return df


def move_completed_files(input_path, completed_path):
    """
    Move processed PDFs to the completed folder and clean up temp page images.

    FIX: pages_dir is now derived from input_path (dynamic) instead of the
         old hardcoded r'./input/pages' — which broke when PDFs were loaded
         from a constituency subfolder like ./input/Udupi/.
    """
    # Clean up the temporary JPEG pages generated during PDF→Image conversion
    pages_dir = os.path.join(input_path, 'pages')
    if os.path.exists(pages_dir):
        shutil.rmtree(pages_dir)

    os.makedirs(completed_path, exist_ok=True)
    listoffiles = []

    for root, dirs, files in os.walk(input_path):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                listoffiles.append(filename)

    for pdf_name in listoffiles:
        src = os.path.join(input_path, pdf_name)
        dst = os.path.join(completed_path, os.path.basename(pdf_name))
        try:
            shutil.move(src, dst)
        except Exception:
            try:
                shutil.rmtree(src)
            except Exception:
                if os.path.isfile(src):
                    os.remove(src)