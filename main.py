#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Karnataka Voter List Extractor — Main Pipeline
===============================================
Usage
─────
  # Download constituency PDFs from Google Drive and extract:
  python main.py --gdrive-input <FOLDER_ID> --constituency Udupi --output ./output --workers 3

  # Process PDFs already in a local folder:
  python main.py --input ./input/Udupi --constituency Udupi --output ./output

  # List all constituency folders in a Drive folder (no extraction):
  python main.py --gdrive-input <FOLDER_ID> --list-constituencies

Google Drive one-time setup
────────────────────────────
  1. https://console.cloud.google.com/
     → Select / create a project
     → "Enable APIs" → enable "Google Drive API"
  2. "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
     Application type: Desktop app
     → Download the JSON → save as credentials.json (same folder as script)
  3. pip install google-api-python-client google-auth-oauthlib google-auth-httplib2
  4. First run opens a browser consent window; token.json is cached after that.
"""

from __future__ import annotations

import os
import sys
import json
import base64
import time
import re as _re
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from datetime import datetime

import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logic
from config import BaseConfig

# ── CRITICAL FIX: force unbuffered stdout ────────────────────────────────────
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass  # Python < 3.7

import builtins as _builtins
_orig_print = _builtins.print
def _flushing_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)
_builtins.print = _flushing_print


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE STOPWATCH
# ═══════════════════════════════════════════════════════════════════════════════

class _Stopwatch:
    """Lightweight phase timer with inline prints and end-of-run summary."""

    def __init__(self):
        self._phases    = []
        self._cur_name  = None
        self._cur_start = None
        self._wall      = time.perf_counter()

    @staticmethod
    def _fmt(s):
        if s < 60:
            return f"{s:.1f}s"
        m, sec = divmod(s, 60)
        return f"{int(m)}m {sec:.0f}s"

    @staticmethod
    def _now_str():
        return datetime.now().strftime("%H:%M:%S")

    def start(self, name: str):
        if self._cur_start is not None:
            self._stop_current()
        self._cur_name  = name
        self._cur_start = time.perf_counter()
        print(f"[{self._now_str()}]  ▶  {name}")

    def stop(self):
        self._stop_current()

    def _stop_current(self):
        if self._cur_start is None:
            return
        elapsed = time.perf_counter() - self._cur_start
        ts = self._now_str()
        self._phases.append((self._cur_name, elapsed, ts))
        print(f"[{ts}]  ✓  {self._cur_name} — {self._fmt(elapsed)}")
        self._cur_name = self._cur_start = None
        return elapsed

    def total_elapsed(self):
        return time.perf_counter() - self._wall

    def summary(self) -> str:
        if not self._phases:
            return "(no phases recorded)"
        col_w = max(len(p[0]) for p in self._phases) + 2
        sep   = "─" * (col_w + 32)
        lines = [
            "",
            "═" * (col_w + 32),
            f"  EXTRACTION TIMING SUMMARY   {datetime.now().strftime('%d-%b-%Y')}",
            "═" * (col_w + 32),
            f"  {'Phase':<{col_w}} {'Duration':>10}  {'Cumulative':>12}",
            sep,
        ]
        cumul = 0.0
        for name, sec, _ in self._phases:
            cumul += sec
            lines.append(
                f"  {name:<{col_w}} {self._fmt(sec):>10}  {self._fmt(cumul):>12}"
            )
        lines += [
            sep,
            f"  {'TOTAL':.<{col_w}} {self._fmt(self.total_elapsed()):>10}",
            "═" * (col_w + 32),
            "",
        ]
        return "\n".join(lines)

    def save_log(self, output_path: str, pdf_list: list):
        try:
            os.makedirs(output_path, exist_ok=True)
            log_file = os.path.join(
                output_path,
                "timing_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".txt"
            )
            header = (
                f"Extraction Timing Log\n"
                f"Run started : {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}\n"
                f"PDFs        : {', '.join(pdf_list) if pdf_list else 'none'}\n"
            )
            with open(log_file, "w", encoding="utf-8") as fh:
                fh.write(header + self.summary())
            print(f"  Timing log saved → {log_file}")
        except Exception as _e:
            print(f"  (could not save timing log: {_e})")


# ═══════════════════════════════════════════════════════════════════════════════
#  GROQ VISION FALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

_VOTER_ID_STRICT_RE = _re.compile(r'^[A-Z]{2,4}\d{5,7}$')

_GROQ_PROMPT = (
    "This is a page from an Indian voter electoral roll (Karnataka, English).\n"
    "Voter cards are arranged in a 3-column × 5-row grid (up to 15 cards per page).\n"
    "Each card has a serial number in its top-left corner (1, 2, 3 …).\n"
    "The LAST printed line of every card reads:\n"
    "  Age : <number>   Gender : Male    OR    Age : <number>   Gender : Female\n\n"
    "Extract the age (integer) and gender (Male / Female) for EVERY card visible.\n"
    "Return ONLY a JSON array — no markdown, no explanation:\n"
    '[{"serial_no":1,"age":45,"gender":"Male"},{"serial_no":2,"age":32,"gender":"Female"},...]\n'
    "Use null for any field you cannot read clearly."
)

_GROQ_VOTERID_PROMPT = (
    "This is a page from an Indian voter electoral roll (Karnataka, English).\n"
    "Voter cards are arranged in a 3-column × 5-row grid (up to 15 cards per page).\n"
    "Each card has a serial number in its top-left corner (1, 2, 3 …).\n"
    "Near the TOP of each card is a Voter ID / EPIC number — a code like\n"
    "  STN2024982   or   LWZ1014919\n"
    "made of 2-4 uppercase letters immediately followed by 5-7 digits.\n\n"
    "Read the Voter ID for EVERY card visible, exactly as printed.\n"
    "Return ONLY a JSON array — no markdown, no explanation:\n"
    '[{"serial_no":1,"voter_id":"STN2024982"},{"serial_no":2,"voter_id":"LWZ1014919"},...]\n'
    "Use null for any card whose Voter ID you cannot read clearly."
)


def _is_incomplete_voter_id(v) -> bool:
    """True if v is empty/NaN OR present but not matching the strict voter-ID pattern."""
    if pd.isna(v):
        return True
    s = str(v).strip()
    return not bool(_VOTER_ID_STRICT_RE.match(s)) if s else True


def _row_is_incomplete(row) -> bool:
    """True if any key field is missing in a voter row."""
    def _miss(v):
        return pd.isna(v) or str(v).strip() in ('', 'not found', 'not available')
    return (_is_incomplete_voter_id(row.get('voterid'))
            or _miss(row.get('age'))
            or _miss(row.get('gender'))
            or _miss(row.get('name'))
            or _miss(row.get('address')))


def _groq_age_gender_fallback(export_df, input_dir, groq_api_key,
                               max_workers=3, rpm=25):
    """Groq Vision fallback: fill age/gender for rows where Tesseract failed."""
    try:
        from groq import Groq
    except ImportError:
        print("  [Groq] 'groq' package not found — run: pip install groq")
        return export_df

    missing_mask = export_df['age'].isna() | export_df['gender'].isna()
    if not missing_mask.any():
        return export_df

    print(f"  [Groq] {int(missing_mask.sum())} rows with missing age/gender…")

    pages_dir = os.path.join(input_dir, 'pages')
    if not os.path.isdir(pages_dir):
        print(f"  [Groq] pages directory not found ({pages_dir}) — skipping.")
        return export_df

    _lock       = Lock()
    _call_times = []

    def _rate_limit():
        with _lock:
            now = time.time()
            _call_times[:] = [t for t in _call_times if now - t < 60.0]
            if len(_call_times) >= rpm:
                sleep_for = (_call_times[0] + 60.0) - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
            _call_times.append(time.time())

    export_df = export_df.copy()
    _card_pos = {}
    for sv in export_df['split'].dropna().unique():
        for pos, idx in enumerate(export_df[export_df['split'] == sv].index):
            _card_pos[idx] = pos

    def _serial_no(idx, split_val):
        sv  = str(split_val).rstrip('.')
        col = int(sv.split('-')[-1]) if sv.split('-')[-1].isdigit() else 0
        return _card_pos.get(idx, 0) * 3 + col + 1

    export_df['_sn'] = [
        _serial_no(i, r['split']) for i, r in export_df.iterrows()
    ]

    pages_needed = export_df.loc[missing_mask, 'page'].dropna().unique().tolist()
    print(f"  [Groq] sending {len(pages_needed)} page image(s) — "
          f"{max_workers} workers, {rpm} RPM limit…")

    client = Groq(api_key=groq_api_key)

    def _call_groq(page_val):
        img_path = os.path.join(pages_dir, str(page_val).strip() + '.jpg')
        if not os.path.isfile(img_path):
            return page_val, {}
        _rate_limit()
        try:
            with open(img_path, 'rb') as fh:
                b64 = base64.b64encode(fh.read()).decode()
            resp = client.chat.completions.create(
                model='meta-llama/llama-4-scout-17b-16e-instruct',
                messages=[{'role': 'user', 'content': [
                    {'type': 'text',      'text': _GROQ_PROMPT},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
                ]}],
                temperature=0.0, max_tokens=1000
            )
            raw = resp.choices[0].message.content.strip()
            raw = _re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', raw, flags=_re.MULTILINE).strip()
            cards = json.loads(raw) if raw.startswith('[') else []
            result = {}
            for c in (cards if isinstance(cards, list) else []):
                sn = c.get('serial_no')
                if sn is None:
                    continue
                try:
                    av = int(c.get('age'))
                    age_s = str(av) if 1 <= av <= 110 else None
                except (TypeError, ValueError):
                    age_s = None
                g = str(c.get('gender') or '').strip()
                gen_s = g.capitalize() if g.lower() in ('male', 'female') else None
                result[int(sn)] = (age_s, gen_s)
            return page_val, result
        except Exception as exc:
            print(f"  [Groq] error on {page_val}: {exc}")
            return page_val, {}

    page_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_call_groq, p): p for p in pages_needed}
        for fut in as_completed(futs):
            pv, res = fut.result()
            page_results[pv] = res

    filled_a = filled_g = 0
    for idx, row in export_df.iterrows():
        a_miss = pd.isna(row['age'])
        g_miss = pd.isna(row['gender'])
        if not (a_miss or g_miss):
            continue
        sn_val    = int(row['_sn'])
        a_s, g_s  = page_results.get(row['page'], {}).get(sn_val, (None, None))
        if a_miss and a_s:
            export_df.at[idx, 'age']    = a_s;  filled_a += 1
        if g_miss and g_s:
            export_df.at[idx, 'gender'] = g_s;  filled_g += 1

    export_df.drop(columns=['_sn'], inplace=True, errors='ignore')
    print(f"  [Groq] ✓ filled {filled_a} ages + {filled_g} genders.")
    return export_df


def _groq_voterid_fallback(export_df, input_dir, groq_api_key,
                            max_workers=3, rpm=25):
    """Groq Vision fallback: fills EMPTY voter IDs and repairs INCOMPLETE ones."""
    try:
        from groq import Groq
    except ImportError:
        print("  [Groq] 'groq' package not found — run: pip install groq")
        return export_df

    incomplete_mask = export_df['voterid'].apply(_is_incomplete_voter_id)
    if not incomplete_mask.any():
        return export_df

    print(f"  [Groq] {int(incomplete_mask.sum())} rows with empty/incomplete voter ID…")
    pages_dir = os.path.join(input_dir, 'pages')
    if not os.path.isdir(pages_dir):
        print(f"  [Groq] pages directory not found ({pages_dir}) — skipping.")
        return export_df

    _lock       = Lock()
    _call_times = []

    def _rate_limit():
        with _lock:
            now = time.time()
            _call_times[:] = [t for t in _call_times if now - t < 60.0]
            if len(_call_times) >= rpm:
                sleep_for = (_call_times[0] + 60.0) - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
            _call_times.append(time.time())

    export_df = export_df.copy()
    _card_pos = {}
    for sv in export_df['split'].dropna().unique():
        for pos, idx in enumerate(export_df[export_df['split'] == sv].index):
            _card_pos[idx] = pos

    def _serial_no(idx, split_val):
        sv  = str(split_val).rstrip('.')
        col = int(sv.split('-')[-1]) if sv.split('-')[-1].isdigit() else 0
        return _card_pos.get(idx, 0) * 3 + col + 1

    export_df['_sn'] = [_serial_no(i, r['split']) for i, r in export_df.iterrows()]

    pages_needed = export_df.loc[incomplete_mask, 'page'].dropna().unique().tolist()
    print(f"  [Groq] sending {len(pages_needed)} page image(s)…")

    client = Groq(api_key=groq_api_key)

    def _call_groq(page_val):
        img_path = os.path.join(pages_dir, str(page_val).strip() + '.jpg')
        if not os.path.isfile(img_path):
            return page_val, {}
        _rate_limit()
        try:
            with open(img_path, 'rb') as fh:
                b64 = base64.b64encode(fh.read()).decode()
            resp = client.chat.completions.create(
                model='meta-llama/llama-4-scout-17b-16e-instruct',
                messages=[{'role': 'user', 'content': [
                    {'type': 'text',      'text': _GROQ_VOTERID_PROMPT},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
                ]}],
                temperature=0.0, max_tokens=1000
            )
            raw = resp.choices[0].message.content.strip()
            raw = _re.sub(r'^```(?:json)?\s*|\s*```\s*$', '', raw, flags=_re.MULTILINE).strip()
            cards = json.loads(raw) if raw.startswith('[') else []
            result = {}
            for c in (cards if isinstance(cards, list) else []):
                sn = c.get('serial_no')
                vid = str(c.get('voter_id') or '').strip().upper()
                if sn and vid and _VOTER_ID_STRICT_RE.match(vid):
                    result[int(sn)] = vid
            return page_val, result
        except Exception as exc:
            print(f"  [Groq] error on {page_val}: {exc}")
            return page_val, {}

    page_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_call_groq, p): p for p in pages_needed}
        for fut in as_completed(futs):
            pv, res = fut.result()
            page_results[pv] = res

    filled_v = 0
    for idx, row in export_df.iterrows():
        if not incomplete_mask.loc[idx]:
            continue
        sn_val = int(row['_sn'])
        new_vid = page_results.get(row['page'], {}).get(sn_val)
        if new_vid:
            export_df.at[idx, 'voterid'] = new_vid
            filled_v += 1

    export_df.drop(columns=['_sn'], inplace=True, errors='ignore')
    print(f"  [Groq] ✓ filled/repaired {filled_v} voter IDs.")
    return export_df


def _azure_di_catchall_fallback(export_df, input_dir, di_endpoint, di_key,
                                 max_workers=3, rpm=15):
    """Azure Document Intelligence — final catch-all for still-incomplete rows."""
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        print("  [Azure DI] package not installed — "
              "run: pip install azure-ai-documentintelligence")
        return export_df

    incomplete_mask = export_df.apply(_row_is_incomplete, axis=1)
    if not incomplete_mask.any():
        return export_df

    n_inc = int(incomplete_mask.sum())
    print(f"  [Azure DI] {n_inc} rows still incomplete — calling Document Intelligence…")

    pages_dir = os.path.join(input_dir, 'pages')
    if not os.path.isdir(pages_dir):
        print(f"  [Azure DI] pages directory not found ({pages_dir}) — skipping.")
        return export_df

    client = DocumentIntelligenceClient(di_endpoint, AzureKeyCredential(di_key))

    export_df = export_df.copy()
    _card_pos = {}
    for sv in export_df['split'].dropna().unique():
        for pos, idx in enumerate(export_df[export_df['split'] == sv].index):
            _card_pos[idx] = pos

    def _serial_no(idx, split_val):
        sv  = str(split_val).rstrip('.')
        col = int(sv.split('-')[-1]) if sv.split('-')[-1].isdigit() else 0
        return _card_pos.get(idx, 0) * 3 + col + 1

    export_df['_sn'] = [_serial_no(i, r['split']) for i, r in export_df.iterrows()]

    pages_needed = export_df.loc[incomplete_mask, 'page'].dropna().unique().tolist()
    print(f"  [Azure DI] analysing {len(pages_needed)} page image(s)…")

    _lock       = Lock()
    _call_times = []

    def _rate_limit():
        with _lock:
            now = time.time()
            _call_times[:] = [t for t in _call_times if now - t < 60.0]
            if len(_call_times) >= rpm:
                sleep_for = (_call_times[0] + 60.0) - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
            _call_times.append(time.time())

    def _bucket_lines_to_cards(result, page_w, page_h):
        """Group lines by estimated card position on the page."""
        cards = {}
        for p in result.pages:
            if not p.lines:
                continue
            for line in p.lines:
                if not line.polygon or len(line.polygon) < 2:
                    continue
                x0 = line.polygon[0] / page_w
                y0 = line.polygon[1] / page_h
                col = min(2, int(x0 * 3))
                row = min(4, int(y0 * 5))
                sn  = row * 3 + col + 1
                cards.setdefault(sn, []).append(line.content)
        return cards

    def _parse_card_text(lines):
        text = ' '.join(lines)
        result = {}
        vid_m = _re.search(r'\b[A-Z]{2,4}\d{5,7}\b', text.upper())
        if vid_m and _VOTER_ID_STRICT_RE.match(vid_m.group()):
            result['voterid'] = vid_m.group()
        age_m = _re.search(r'Age\s*[:\-]?\s*(\d{1,3})', text, _re.IGNORECASE)
        if age_m:
            result['age'] = age_m.group(1)
        gen_m = _re.search(r'Gender\s*[:\-]?\s*(Male|Female)', text, _re.IGNORECASE)
        if gen_m:
            result['gender'] = gen_m.group(1).capitalize()
        name_m = _re.search(r'Name\s*[:\-]\s*(.+?)(?:\s{2,}|$)', text, _re.IGNORECASE)
        if name_m:
            result['name'] = name_m.group(1).strip()
        house_m = _re.search(r'House[^:\-]*[:\-]\s*(.+?)(?:\s{2,}|$)', text, _re.IGNORECASE)
        if house_m:
            result['address'] = house_m.group(1).strip()
        return result

    def _call_azure(page_val):
        img_path = os.path.join(pages_dir, str(page_val).strip() + '.jpg')
        if not os.path.isfile(img_path):
            return page_val, {}
        _rate_limit()
        try:
            with open(img_path, 'rb') as fh:
                img_bytes = fh.read()
            poller = client.begin_analyze_document(
                "prebuilt-layout",
                AnalyzeDocumentRequest(bytes_source=img_bytes),
            )
            result = poller.result()
            if not result.pages:
                return page_val, {}
            page0  = result.pages[0]
            page_w = page0.width  or 1
            page_h = page0.height or 1
            cards_lines  = _bucket_lines_to_cards(result, page_w, page_h)
            cards_parsed = {sn: _parse_card_text(lines)
                            for sn, lines in cards_lines.items() if lines}
            return page_val, cards_parsed
        except Exception as exc:
            print(f"  [Azure DI] error on {page_val}: {exc}")
            return page_val, {}

    page_results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_call_azure, p): p for p in pages_needed}
        for fut in as_completed(futs):
            pv, res = fut.result()
            page_results[pv] = res

    filled_counts = {'voterid': 0, 'age': 0, 'gender': 0, 'name': 0, 'address': 0}
    for idx, row in export_df.iterrows():
        if not incomplete_mask.loc[idx]:
            continue
        sn_val  = int(row['_sn'])
        di_card = page_results.get(row['page'], {}).get(sn_val)
        if not di_card:
            continue
        if _is_incomplete_voter_id(row.get('voterid')) and di_card.get('voterid'):
            export_df.at[idx, 'voterid'] = di_card['voterid']
            filled_counts['voterid'] += 1
        for col in ('age', 'gender', 'name', 'address'):
            cur = row.get(col)
            if (pd.isna(cur) or str(cur).strip() == '') and di_card.get(col):
                export_df.at[idx, col] = di_card[col]
                filled_counts[col] += 1

    export_df.drop(columns=['_sn'], inplace=True, errors='ignore')
    print(f"  [Azure DI] ✓ filled: "
          f"{filled_counts['voterid']} voter IDs, "
          f"{filled_counts['age']} ages, "
          f"{filled_counts['gender']} genders, "
          f"{filled_counts['name']} names, "
          f"{filled_counts['address']} addresses.")
    return export_df


# ═══════════════════════════════════════════════════════════════════════════════
#  GOOGLE DRIVE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_GD_SCOPES      = ["https://www.googleapis.com/auth/drive"]
_GD_MIME_FOLDER = "application/vnd.google-apps.folder"
_GD_CHUNK       = 1 * 1024 * 1024    # 1 MB chunks
_GD_MAX_RETRIES = 8
_GD_RETRY_DELAY = 2.0
_GD_TIMEOUT     = 120


def _gd_auth(cred_file: str = "credentials.json", tok_file: str = "token.json"):
    """
    Authenticate via OAuth2 + token cache.
    First call: opens browser for Google account consent.
    Subsequent calls: silently refreshes from token.json.
    Returns a Drive v3 service object.
    """
    try:
        from googleapiclient.discovery import build as _gdrive_build
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request as _OAuthRequest
    except ImportError:
        sys.exit(
            "❌  Google Drive packages not installed.\n"
            "    Run: pip install google-api-python-client "
            "google-auth-oauthlib google-auth-httplib2"
        )

    creds = None
    if os.path.exists(tok_file):
        creds = Credentials.from_authorized_user_file(tok_file, _GD_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(_OAuthRequest())
        else:
            if not os.path.exists(cred_file):
                sys.exit(
                    f"❌  '{cred_file}' not found.\n"
                    "    Download from: Google Cloud Console → APIs & Services\n"
                    "    → Credentials → Create OAuth 2.0 Client ID (Desktop app)"
                )
            flow  = InstalledAppFlow.from_client_secrets_file(cred_file, _GD_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(tok_file, "w") as fh:
            fh.write(creds.to_json())

    from googleapiclient.discovery import build as _gdrive_build
    return _gdrive_build("drive", "v3", credentials=creds)


def _gd_ls_folders(svc, parent_id: str) -> list:
    """Return [{id, name}] for every subfolder in parent_id."""
    items, tok = [], None
    while True:
        kw = dict(
            q=f"'{parent_id}' in parents and "
              f"mimeType='{_GD_MIME_FOLDER}' and trashed=false",
            fields="nextPageToken, files(id, name)",
            orderBy="name", pageSize=200,
        )
        if tok:
            kw["pageToken"] = tok
        r   = svc.files().list(**kw).execute()
        items.extend(r.get("files", []))
        tok = r.get("nextPageToken")
        if not tok:
            break
    return items


def _gd_ls_pdfs(svc, folder_id: str) -> list:
    """Return [{id, name, size}] for every PDF in folder_id."""
    items, tok = [], None
    while True:
        kw = dict(
            q=f"'{folder_id}' in parents and "
              f"mimeType='application/pdf' and trashed=false",
            fields="nextPageToken, files(id, name, size)",
            orderBy="name", pageSize=200,
        )
        if tok:
            kw["pageToken"] = tok
        r   = svc.files().list(**kw).execute()
        items.extend(r.get("files", []))
        tok = r.get("nextPageToken")
        if not tok:
            break
    return items


def _gd_download(svc, file_id: str, local_path: str) -> None:
    """
    Download a Drive file to local_path with:
      • Skip      — if file already exists with non-zero size
      • Retry     — up to _GD_MAX_RETRIES attempts with exponential back-off
      • Timeout   — per-chunk socket timeout
    """
    import socket
    import gc
    try:
        from googleapiclient.http import MediaIoBaseDownload
        from googleapiclient.errors import HttpError as _HttpError
    except ImportError:
        raise RuntimeError("google-api-python-client not installed")

    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return   # already downloaded — skip

    fname = os.path.basename(local_path)
    gc.collect()

    req  = svc.files().get_media(fileId=file_id)

    with open(local_path, "wb") as fh:
        chunk = _GD_CHUNK
        dl    = MediaIoBaseDownload(fh, req, chunksize=chunk)
        done  = False

        while not done:
            last_exc = None
            for attempt in range(1, _GD_MAX_RETRIES + 1):
                try:
                    import httplib2
                    old_timeout = socket.getdefaulttimeout()
                    socket.setdefaulttimeout(_GD_TIMEOUT)
                    try:
                        _, done = dl.next_chunk()
                    finally:
                        socket.setdefaulttimeout(old_timeout)
                    last_exc = None
                    break

                except MemoryError as exc:
                    last_exc = exc
                    gc.collect()
                    chunk = max(256 * 1024, chunk // 2)
                    dl = MediaIoBaseDownload(fh, req, chunksize=chunk)
                    time.sleep(1)

                except (TimeoutError, socket.timeout, socket.error,
                        ConnectionResetError, ConnectionError, OSError) as exc:
                    last_exc = exc
                    wait = _GD_RETRY_DELAY * (2 ** (attempt - 1))
                    print(f"    ⚠  [{fname}] network error (attempt {attempt}/"
                          f"{_GD_MAX_RETRIES}): {type(exc).__name__} — "
                          f"retrying in {wait:.0f}s…")
                    time.sleep(wait)

            if last_exc is not None:
                fh.close()
                try:
                    os.remove(local_path)
                except OSError:
                    pass
                raise RuntimeError(
                    f"❌  Failed to download '{fname}' after {_GD_MAX_RETRIES} retries. "
                    f"Last error: {last_exc}"
                ) from last_exc


def download_constituency_pdfs(svc, gdrive_input_id: str,
                                constituency_name: str,
                                local_input_root: str) -> str:
    """
    Find the constituency subfolder in the Drive folder, download all its PDFs
    to local_input_root/<constituency_name>/ and return the local directory path.

    Raises SystemExit if the constituency folder is not found.
    """
    print(f"\n🔍  Searching for constituency folder: '{constituency_name}'…")
    sub_folders = _gd_ls_folders(svc, gdrive_input_id)
    if not sub_folders:
        sys.exit(f"❌  No subfolders found in Drive folder ID: {gdrive_input_id}")

    match = next(
        (f for f in sub_folders if f["name"].lower() == constituency_name.lower()),
        None
    )
    if match is None:
        available = ", ".join(f["name"] for f in sub_folders)
        sys.exit(
            f"❌  Constituency '{constituency_name}' not found in Drive folder.\n"
            f"    Available folders: {available}\n"
            f"    Tip: use --list-constituencies to see all available names."
        )

    cid    = match["id"]
    cname  = match["name"]   # use Drive's exact casing
    print(f"  ✅  Found: '{cname}'  (id={cid})")

    pdf_files = _gd_ls_pdfs(svc, cid)
    if not pdf_files:
        sys.exit(f"❌  No PDF files found in Drive folder '{cname}'.")

    local_dir = os.path.join(local_input_root, cname)
    os.makedirs(local_dir, exist_ok=True)

    print(f"\n⬇️   Downloading {len(pdf_files)} PDF(s) → {local_dir}")
    for pf in pdf_files:
        # Strip "Copy of " prefixes that Drive sometimes adds on shared files
        safe_name = _re.sub(r"^(?:Copy\s+of\s+)+", "", pf["name"], flags=_re.I)
        dest = os.path.join(local_dir, safe_name)
        print(f"  {'↳':>4}  {safe_name}  ({int(pf.get('size', 0)) // 1024:,} KB)")
        _gd_download(svc, pf["id"], dest)

    print(f"  ✅  {len(pdf_files)} PDFs ready → {local_dir}\n")
    return local_dir


def list_constituencies(svc, gdrive_input_id: str) -> None:
    """Print all constituency folder names found in the Drive input folder."""
    sub_folders = _gd_ls_folders(svc, gdrive_input_id)
    if not sub_folders:
        print("  (no subfolders found)")
        return
    print(f"\n📂  Constituency folders in Drive ({len(sub_folders)} found):")
    for f in sub_folders:
        pdf_count = len(_gd_ls_pdfs(svc, f["id"]))
        print(f"     • {f['name']}  ({pdf_count} PDFs)")


# ═══════════════════════════════════════════════════════════════════════════════
#  CASTE / SUB-CASTE ENRICHMENT  (from test.py logic, inlined here)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_caste_enrichment(export_df: pd.DataFrame, caste_file_path: str) -> pd.DataFrame:
    """
    Add 'sub_caste' and 'caste' columns by matching voter/father names against
    the caste Excel file.  Returns the enriched DataFrame.

    If the caste file doesn't exist or fails to load, the original DataFrame
    is returned unchanged (non-fatal).
    """
    if not os.path.isfile(caste_file_path):
        print(f"  [Caste] File not found: {caste_file_path} — skipping enrichment.")
        return export_df

    try:
        df = export_df.copy(deep=True)
        df["sub_caste"] = None

        # ── Sub-caste from name sheets ────────────────────────────────────────
        df_excel = pd.ExcelFile(caste_file_path)
        for sheet in df_excel.sheet_names:
            if sheet.lower() == "caste":
                continue
            caste_df = pd.read_excel(caste_file_path, sheet_name=sheet)
            if caste_df.empty or "Names" not in caste_df.columns:
                continue
            caste_list = [str(x).lower() for x in caste_df["Names"] if pd.notna(x)]

            # Match against father name
            for rowcount, name in enumerate(df["father"]):
                if rowcount >= len(df):
                    break
                for cname in caste_list:
                    if cname and cname in str(name).lower():
                        df.at[rowcount, "sub_caste"] = sheet
                        break

            # Match against voter name
            for rowcount, name in enumerate(df["name"]):
                if rowcount >= len(df):
                    break
                if name and not pd.isnull(name):
                    for cname in caste_list:
                        if cname and cname in str(name).lower():
                            df.at[rowcount, "sub_caste"] = sheet
                            break

        # ── Top-level caste from Sub_Caste → Caste mapping ───────────────────
        caste_map_df = pd.read_excel(caste_file_path, sheet_name="Caste")
        caste_map_df.columns = ["Sub_Caste", "caste"]
        df = pd.merge(df, caste_map_df, how="left",
                      left_on="sub_caste", right_on="Sub_Caste")
        # Drop the duplicate 'Sub_Caste' column from the merge
        if "Sub_Caste" in df.columns:
            df.drop(columns=["Sub_Caste"], inplace=True)

        print(f"  [Caste] Enrichment complete: "
              f"{df['sub_caste'].notna().sum()} rows tagged.")
        return df

    except Exception as exc:
        print(f"  [Caste] Error during enrichment (non-fatal): {exc}")
        return export_df


# ═══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="Karnataka Voter List Extractor — Google Drive + Local mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GDrive mode — download Udupi PDFs and extract:
  python main.py --gdrive-input 1eFf9lKcAei3... --constituency Udupi --output ./output

  # GDrive mode — list all constituency folders:
  python main.py --gdrive-input 1eFf9lKcAei3... --list-constituencies

  # Local mode — process PDFs already in ./input/Udupi/:
  python main.py --input ./input/Udupi --constituency Udupi --output ./output

  # With caste enrichment:
  python main.py --gdrive-input 1eFf9lKcAei3... --constituency Udupi \\
                 --output ./output --caste-file ./caste.xlsx
        """
    )

    # ── Input source ──────────────────────────────────────────────────────────
    src = ap.add_argument_group("Input source (choose one)")
    src.add_argument(
        "--gdrive-input", metavar="FOLDER_ID", default=None,
        help="Google Drive folder ID that contains constituency subfolders. "
             "Get it from the URL: drive.google.com/drive/folders/<FOLDER_ID>"
    )
    src.add_argument(
        "--input", "-i", metavar="DIR", default=None,
        help="Local directory containing PDFs (used when --gdrive-input is not set). "
             f"Defaults to {BaseConfig.input_dir}/<constituency>"
    )

    # ── Constituency ──────────────────────────────────────────────────────────
    ap.add_argument(
        "--constituency", "-c", metavar="NAME", default=None,
        help="Constituency folder name to process (e.g. Udupi, Bantwal). "
             "Required unless --list-constituencies is used."
    )
    ap.add_argument(
        "--list-constituencies", action="store_true",
        help="List all constituency folders in the Drive folder and exit. "
             "Requires --gdrive-input."
    )

    # ── Output ────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--output", "-o", metavar="DIR", default=BaseConfig.output_path,
        help=f"Root output directory. Excel files are saved under "
             f"<output>/<constituency>/. Default: {BaseConfig.output_path}"
    )
    ap.add_argument(
        "--completed", metavar="DIR", default=BaseConfig.completed_path,
        help=f"Directory where processed PDFs are moved after extraction. "
             f"Default: {BaseConfig.completed_path}"
    )

    # ── Google Drive auth ─────────────────────────────────────────────────────
    gd = ap.add_argument_group("Google Drive authentication")
    gd.add_argument(
        "--credentials", metavar="FILE",
        default=BaseConfig.credentials_file,
        help=f"Path to OAuth2 credentials JSON (default: {BaseConfig.credentials_file})"
    )
    gd.add_argument(
        "--gdrive-token", metavar="FILE",
        default=BaseConfig.gdrive_token_file,
        help=f"Cached Drive auth token path (default: {BaseConfig.gdrive_token_file}). "
             "Created automatically on first run."
    )

    # ── Optional enrichment ───────────────────────────────────────────────────
    ap.add_argument(
        "--caste-file", metavar="FILE", default=None,
        help="Path to caste.xlsx for sub-caste enrichment. "
             "Skipped if not provided or file not found."
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--workers", "-w", type=int, default=3,
        help="Number of parallel OCR workers (default: 3)"
    )
    ap.add_argument(
        "--keep-input", action="store_true",
        help="Do not move processed PDFs to the completed folder after extraction."
    )
    ap.add_argument(
        "--force-download", action="store_true",
        help="Always download from Drive even if PDFs already exist locally. "
             "By default the download is skipped when the local folder already "
             "contains PDF files (e.g. you uploaded them manually)."
    )
    ap.add_argument(
        "--fresh-start", action="store_true",
        help="Ignore any existing checkpoint and re-extract ALL PDFs from scratch. "
             "By default the extractor automatically resumes from where it stopped "
             "(resume is always on, no extra flag needed)."
    )

    return ap.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  PER-PDF RUNNING SAVE
# ═══════════════════════════════════════════════════════════════════════════════

# Columns written to the running Excel — same order as the final output.
_EXPORT_COLS = [
    'id', 'page', 'split',
    'assembly_constituency_no', 'assembly_constituency_name',
    'section_no_and_name', 'part_no',
    'polling station', 'polling address',
    'voterid', 'name', 'Relative Name', 'Relation Type',
    'address', 'age', 'gender',
]

_VID_OK_RE = _re.compile(r'^[A-Z]{2,3}\d{6,8}$')


def _quick_save_running(df3: pd.DataFrame,
                         constituency_out: str,
                         constituency: str,
                         completed: int,
                         total: int) -> None:
    """
    Save a cumulative snapshot of all rows extracted so far to a single
    running Excel file.  Called after every PDF completes.

    • Overwrites the same file each time  →  open it at any point to see
      current state without waiting for the full 227-PDF run to finish.
    • Applies only the fast/essential cleanup so the file is immediately
      readable (ghost-row drop, invalid voter-ID clear, gender normalise).
      Full cleanup (Groq/Azure fallbacks, prefix repair …) still runs at
      the end on the master file.
    • Non-fatal: if the save fails (e.g. file is open in Excel) a warning
      is printed and extraction continues normally.
    """
    try:
        # ── Select available columns (safe against schema differences) ────────
        cols = [c for c in _EXPORT_COLS if c in df3.columns]
        snap = df3[cols].copy()
        snap.reset_index(drop=True, inplace=True)

        # ── Quick cleanup 1: drop ghost rows (no name AND no valid voter ID) ──
        snap = snap[~(
            (snap['name'].isin(['not found', '']) | snap['name'].isna()) &
            (snap['voterid'].isna() |
             ~snap['voterid'].astype(str).str.match(_VID_OK_RE.pattern, na=False))
        )].reset_index(drop=True)

        # ── Quick cleanup 2: null out obviously invalid voter IDs ─────────────
        bad = (snap['voterid'].notna() &
               ~snap['voterid'].astype(str).str.match(_VID_OK_RE.pattern, na=False))
        snap.loc[bad, 'voterid'] = None

        # ── Quick cleanup 3: normalise Gender OCR variants ────────────────────
        def _fix_g(g):
            if pd.isna(g): return g
            v = _re.sub(r'\W', '', str(g)).strip()
            if _re.fullmatch(r'Fe[a-z]+', v, _re.I): return 'Female'
            if _re.fullmatch(r'Male',     v, _re.I): return 'Male'
            return v or None
        snap['gender'] = snap['gender'].apply(_fix_g)

        # ── Quick cleanup 4: strip relation-type prefix from Relative Name ─────
        # e.g. "Father's Name : Narayana Shetty"  →  "Narayana Shetty"
        # The old regex in logic.py was broken for "Father's" (apostrophe).
        def _strip_rel_prefix(v):
            if pd.isna(v) or str(v).strip() == '': return v
            m = _re.search(r'\bName\s*[:\-]\s*(.+)$', str(v), _re.IGNORECASE)
            return m.group(1).strip() if m else str(v).strip()
        if 'Relative Name' in snap.columns:
            snap['Relative Name'] = snap['Relative Name'].apply(_strip_rel_prefix)

        # ── Quick cleanup 5: replace "not found" name with None ───────────────
        # Keeps the row (voter ID is valid) but leaves name blank rather than
        # showing "not found" as actual data in the running preview.
        if 'name' in snap.columns:
            snap['name'] = snap['name'].replace({'not found': None, 'not available': None})

        # ── Save ──────────────────────────────────────────────────────────────
        running_path = os.path.join(constituency_out,
                                    f"{constituency}_running.xlsx")
        snap.to_excel(running_path, index=False)

        pct = int(completed / total * 100)
        print(f"  💾  Saved [{completed}/{total}  {pct}%]  "
              f"{len(snap):,} rows → {running_path}")

    except PermissionError:
        print(f"  ⚠️   Running save skipped — close the Excel file first "
              f"(file is open in another application).")
    except Exception as exc:
        print(f"  ⚠️   Running save failed (non-fatal): {exc}")



# ═══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT / RESUME  ─  auto-resume always on, no extra flag needed
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Two hidden files are kept inside the constituency output folder:
#
#    .voter_ckpt.json   ─  list of PDF basenames that finished successfully
#    .voter_raw.csv     ─  raw OCR rows (all df3 columns) from those PDFs
#
#  On every re-run:
#    • checkpoint is read  → already-done PDFs are skipped
#    • raw CSV is loaded   → df3 is pre-populated with the old rows
#    • extraction resumes  → only pending PDFs are processed
#
#  Use --fresh-start to wipe the checkpoint and re-extract everything.
# ═══════════════════════════════════════════════════════════════════════════════

def _ckpt_path(out: str) -> str:
    return os.path.join(out, ".voter_ckpt.json")

def _raw_csv_path(out: str) -> str:
    return os.path.join(out, ".voter_raw.csv")


def _ckpt_load(out: str):
    """
    Load checkpoint.
    Returns (done_set, df_prev) where:
      done_set  – set of PDF basenames already fully extracted
      df_prev   – DataFrame of previously extracted rows (None if nothing saved)
    """
    p_ckpt = _ckpt_path(out)
    p_raw  = _raw_csv_path(out)
    done: set = set()
    df_prev   = None

    if not os.path.exists(p_ckpt):
        return done, df_prev          # no checkpoint — fresh run

    try:
        with open(p_ckpt, encoding="utf-8") as f:
            data = json.load(f)
        done    = set(data.get("completed_pdfs", []))
        updated = data.get("updated", "unknown")
        print(f"  📌  Checkpoint found: {len(done)} PDF(s) already done  (saved {updated})")
    except Exception as exc:
        print(f"  ⚠️   Could not read checkpoint (starting fresh): {exc}")
        return set(), None

    if done and os.path.exists(p_raw):
        try:
            df_prev = pd.read_csv(p_raw, low_memory=False, encoding="utf-8-sig")
            print(f"  ✅  Reloaded {len(df_prev):,} previously extracted rows")
        except Exception as exc:
            print(f"  ⚠️   Could not reload previous rows ({exc}) — will re-extract all")
            done    = set()
            df_prev = None

    return done, df_prev


def _ckpt_save(out: str, done: set) -> None:
    """Write the updated set of completed PDF names to the checkpoint JSON."""
    try:
        with open(_ckpt_path(out), "w", encoding="utf-8") as f:
            json.dump(
                {"completed_pdfs": sorted(done),
                 "updated":        datetime.now().isoformat()},
                f, indent=2, ensure_ascii=False,
            )
    except Exception as exc:
        print(f"  ⚠️   Checkpoint save failed (non-fatal): {exc}")


def _ckpt_clear(out: str) -> None:
    """Delete checkpoint + raw CSV so the next run starts completely fresh."""
    for p in (_ckpt_path(out), _raw_csv_path(out)):
        try:
            if os.path.exists(p):
                os.remove(p)
                print(f"  🗑️   Removed: {p}")
        except Exception:
            pass


def _raw_append(new_rows: pd.DataFrame, out: str) -> None:
    """
    Append the rows from one PDF to the hidden raw CSV.
    Header is written only on the first call (when the file doesn't exist yet).
    This CSV is for resume only — not meant for user viewing.
    """
    if new_rows is None or new_rows.empty:
        return
    try:
        p = _raw_csv_path(out)
        write_header = not os.path.exists(p)
        new_rows.to_csv(p, mode="a", header=write_header,
                        index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"  ⚠️   Raw CSV append failed (non-fatal): {exc}")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    sw   = _Stopwatch()

    # ── Validate args ─────────────────────────────────────────────────────────
    if args.list_constituencies:
        if not args.gdrive_input:
            sys.exit("❌  --list-constituencies requires --gdrive-input <FOLDER_ID>")
        svc = _gd_auth(args.credentials, args.gdrive_token)
        list_constituencies(svc, args.gdrive_input)
        return

    if not args.constituency:
        sys.exit(
            "❌  --constituency <NAME> is required.\n"
            "    Example: python main.py --gdrive-input <ID> --constituency Udupi\n"
            "    Run with --list-constituencies to see available folder names."
        )

    constituency = args.constituency

    print(f"\n{'═'*64}")
    print(f"  Karnataka Voter List Extractor")
    print(f"  Constituency : {constituency}")
    print(f"  Output       : {args.output}/{constituency}/")
    print(f"{'═'*64}\n")

    # ── Determine input directory ─────────────────────────────────────────────
    #
    # Priority order:
    #   1. --input <DIR>  explicitly given  → use it directly (no Drive)
    #   2. --gdrive-input given AND local folder already has PDFs
    #      AND --force-download NOT set     → skip download, use local folder
    #   3. --gdrive-input given, no local PDFs (or --force-download set)
    #                                       → authenticate + download from Drive
    #   4. Neither flag given               → look for ./input/<constituency>/

    # Resolve the expected local folder path up front — used in cases 2 & 3
    _local_input_root    = args.input or BaseConfig.input_dir
    _expected_local_dir  = (args.input
                            if args.input
                            else os.path.join(BaseConfig.input_dir, constituency))

    def _count_local_pdfs(folder: str) -> int:
        """Return number of .pdf files directly inside folder (non-recursive)."""
        if not os.path.isdir(folder):
            return 0
        return sum(
            1 for f in os.listdir(folder)
            if f.lower().endswith(".pdf") and
               os.path.isfile(os.path.join(folder, f))
        )

    if args.input:
        # ── Case 1: explicit local path ───────────────────────────────────────
        input_dir = args.input
        if not os.path.isdir(input_dir):
            sys.exit(
                f"❌  Input directory not found: {input_dir}"
            )
        n_local = _count_local_pdfs(input_dir)
        print(f"  📂  Using local folder: {input_dir}  ({n_local} PDF(s) found)")

    elif args.gdrive_input:
        n_local = _count_local_pdfs(_expected_local_dir)

        if n_local > 0 and not args.force_download:
            # ── Case 2: local PDFs already present — skip Drive download ──────
            input_dir = _expected_local_dir
            print(f"  ♻️   Local PDFs detected — skipping Drive download.")
            print(f"  📂  Folder : {input_dir}")
            print(f"  📄  PDFs   : {n_local} file(s) ready for extraction")
            print(f"       (use --force-download to re-download from Drive anyway)\n")
        else:
            # ── Case 3: download from Drive ───────────────────────────────────
            if n_local > 0 and args.force_download:
                print(f"  ⚠️   --force-download set — re-downloading "
                      f"over {n_local} existing local PDF(s).")
            sw.start("Google Drive authentication + PDF download")
            svc = _gd_auth(args.credentials, args.gdrive_token)
            print("  ✅  Authenticated with Google Drive")
            input_dir = download_constituency_pdfs(
                svc, args.gdrive_input, constituency,
                local_input_root=BaseConfig.input_dir
            )
            sw.stop()

    else:
        # ── Case 4: no Drive flag, no --input — look for default local folder ─
        input_dir = _expected_local_dir
        if not os.path.isdir(input_dir):
            sys.exit(
                f"❌  Input directory not found: {input_dir}\n"
                f"    Options:\n"
                f"      • Put PDFs in that folder and re-run, OR\n"
                f"      • Use --gdrive-input <FOLDER_ID> to download from Drive, OR\n"
                f"      • Use --input <DIR> to point to a different local folder."
            )
        n_local = _count_local_pdfs(input_dir)
        print(f"  📂  Using local folder: {input_dir}  ({n_local} PDF(s) found)")

    # ── Build output paths for this constituency ──────────────────────────────
    constituency_out  = os.path.join(args.output, constituency)
    completed_path    = os.path.join(args.completed, constituency)
    os.makedirs(constituency_out, exist_ok=True)
    os.makedirs(completed_path,   exist_ok=True)

    # ── Stage 1: PDF → Image conversion + OCR  (with auto-resume) ───────────
    list_of_pdfs = logic.pdfs_identification(input_dir)
    if not list_of_pdfs:
        sys.exit(f"❌  No PDF files found in: {input_dir}")

    total_pdfs   = len(list_of_pdfs)
    running_path = os.path.join(constituency_out, f"{constituency}_running.xlsx")

    # ── Load checkpoint unless --fresh-start ──────────────────────────────────
    if getattr(args, "fresh_start", False):
        print("  🗑️   --fresh-start: clearing checkpoint and re-extracting everything.")
        _ckpt_clear(constituency_out)
        done_pdfs = set()
        df3       = BaseConfig.df3.copy()
    else:
        done_pdfs, df_prev = _ckpt_load(constituency_out)
        df3 = df_prev if df_prev is not None else BaseConfig.df3.copy()

    # ── Separate pending from already-done PDFs ───────────────────────────────
    pending_pdfs = [p for p in list_of_pdfs if p not in done_pdfs]
    n_done       = total_pdfs - len(pending_pdfs)

    print(f"\n  📋  Total PDFs      : {total_pdfs}")
    if n_done:
        print(f"  ✅  Already done    : {n_done}  (skipped — use --fresh-start to redo)")
    print(f"  ⏳  To process now  : {len(pending_pdfs)}")
    print(f"  💾  Running output  : {running_path}")
    print(f"      (updated after every PDF — open at any time to see progress)\n")

    if not pending_pdfs:
        print("  ✅  All PDFs already extracted — jumping to cleanup + export.\n")
    else:
        sw.start(f"PDF → Image conversion + OCR  ({len(pending_pdfs)} PDFs remaining)")
        for i, pdf_name in enumerate(pending_pdfs, 1):
            overall = n_done + i          # real position across all 227
            print(f"\n  ▶  [{overall}/{total_pdfs}]  {pdf_name}")

            new_rows = logic.pageSplit(input_dir, pdf_name, BaseConfig.poppler_path)
            df3 = pd.concat([df3, new_rows], axis=0, ignore_index=True)

            # ── Save checkpoint immediately after each PDF ────────────────────
            done_pdfs.add(pdf_name)
            _ckpt_save(constituency_out, done_pdfs)   # .voter_ckpt.json
            _raw_append(new_rows, constituency_out)    # .voter_raw.csv

            # ── Running Excel for live inspection ─────────────────────────────
            _quick_save_running(df3, constituency_out, constituency,
                                overall, total_pdfs)

        sw.stop()

    df3.reset_index(inplace=True, drop=True)

    # ── Stage 2: Build export DataFrame ───────────────────────────────────────
    sw.start("Building export dataframe")
    print("Building export columns…")
    try:
        export_df = df3[[
            'id', 'page', 'split',
            'assembly_constituency_no', 'assembly_constituency_name',
            'section_no_and_name', 'part_no',
            'polling station', 'polling address',
            'voterid', 'name', 'Relative Name', 'Relation Type',
            'address', 'age', 'gender',
        ]]
        sw.stop()
    except Exception as exc:
        sw.stop()
        print(f"  Error building export dataframe: {exc}")
        export_df = df3.copy()

    # ── Stage 3: Quality cleanup ───────────────────────────────────────────────
    sw.start("Quality cleanup (steps 1-9)")
    try:
        # 1. Drop ghost rows (cover/map/summary pages with no real voter data)
        #    Keep rows that have a valid voter ID even if name is "not found"
        before = len(export_df)
        export_df = export_df[~(
            (export_df['name'].isin(['not found', '']) | export_df['name'].isna()) &
            (export_df['voterid'].isna() |
             export_df['voterid'].isin(['MANGALORE', 'SURATHKAL', '']) |
             ~export_df['voterid'].astype(str).str.match(r'^[A-Z]{2,3}\d{6,8}$', na=False))
        )].reset_index(drop=True)
        dropped = before - len(export_df)
        if dropped:
            print(f"  Removed {dropped} ghost rows")

        # 2. Replace "not found" / "not available" in name with None
        #    (the voter card was read correctly; the name label was just missed by OCR)
        _still_nf = export_df['name'].isin(['not found', 'not available'])
        if _still_nf.sum():
            export_df.loc[_still_nf, 'name'] = None
            print(f"  Blanked {int(_still_nf.sum())} 'not found' name placeholders "
                  f"(voter ID + age still kept)")

        # 3. Strip relation-type prefix from Relative Name column
        #    FIX: the old regex in logic.py used 'Fathers?' which never matched
        #         "Father's" (apostrophe) → 100% of values kept the full prefix.
        #    New approach: grab everything after "Name :" / "Name -".
        def _strip_rel_prefix(v):
            if pd.isna(v) or str(v).strip() == '': return v
            m = _re.search(r'\bName\s*[:\-]\s*(.+)$', str(v), _re.IGNORECASE)
            return m.group(1).strip() if m else str(v).strip()

        before_rn = export_df['Relative Name'].copy()
        export_df['Relative Name'] = export_df['Relative Name'].apply(_strip_rel_prefix)
        rn_fixed = (before_rn.fillna('') != export_df['Relative Name'].fillna('')).sum()
        print(f"  Relative Name prefix stripped: {rn_fixed:,} values cleaned "
              f"(e.g. \"Father's Name : X\" → \"X\")")

        # 4. Clear voter IDs that are still invalid
        _vid_ok = _re.compile(r'^[A-Z]{2,3}\d{6,8}$')
        bad_mask = (export_df['voterid'].notna() &
                    ~export_df['voterid'].astype(str).str.match(_vid_ok.pattern, na=False))
        export_df.loc[bad_mask, 'voterid'] = None
        if bad_mask.sum():
            print(f"  Cleared {bad_mask.sum()} invalid voter IDs")

        # 5. Strip special characters from name columns
        _SPECIAL_RE = _re.compile(r'[^A-Za-z0-9 .\-]')

        def _clean_name_col(val):
            if pd.isna(val) or str(val).strip() in ('not found', 'not available', ''):
                return None   # normalise all sentinel values to None
            cleaned = _SPECIAL_RE.sub('', str(val))
            return _re.sub(r' {2,}', ' ', cleaned).strip() or None

        export_df['name']          = export_df['name'].apply(_clean_name_col)
        export_df['Relative Name'] = export_df['Relative Name'].apply(_clean_name_col)
        print("  Special-char cleanup applied to name & Relative Name")

        # 6. Normalise gender OCR variants
        def _fix_gender(g):
            if pd.isna(g): return g
            v = _re.sub(r'\W', '', str(g)).strip()
            if _re.fullmatch(r'Fe[a-z]+', v, _re.I): return 'Female'
            if _re.fullmatch(r'Male', v, _re.I):     return 'Male'
            return v or None

        export_df['gender'] = export_df['gender'].apply(_fix_gender)

        # 5. Fix voter ID prefixes (2→3 chars OCR drop)
        def _fix_prefix(vid):
            if _re.match(r'^TN[0-9]', str(vid)):  return 'S' + str(vid)
            if _re.match(r'^WZ[0-9]', str(vid)):  return 'L' + str(vid)
            if _re.match(r'^LW\d{8}$', str(vid)): return 'LWZ' + str(vid)[3:]
            return vid

        _before_fix = export_df['voterid'].copy()
        export_df['voterid'] = export_df['voterid'].apply(
            lambda v: _fix_prefix(v) if pd.notna(v) else v)
        _pfx_fixed = (_before_fix.fillna('') != export_df['voterid'].fillna('')).sum()
        if _pfx_fixed:
            print(f"  Fixed {_pfx_fixed} voter ID prefixes (2-letter → 3-letter)")

        # 6. Fill missing age from sibling rows sharing the same voter ID
        if 'age' in export_df.columns:
            _age_lookup = (
                export_df.dropna(subset=['voterid', 'age'])
                .groupby('voterid')['age'].first()
            )
            _missing_age = export_df['age'].isna() & export_df['voterid'].notna()
            export_df.loc[_missing_age, 'age'] = (
                export_df.loc[_missing_age, 'voterid'].map(_age_lookup))
            _filled = _missing_age.sum() - export_df['age'].isna().sum()
            if _filled > 0:
                print(f"  Filled {_filled} missing ages from duplicate voter ID rows")

    except Exception as exc:
        print(f"  Quality cleanup error (non-fatal): {exc}")

    sw.stop()

    # ── Stage 4: Groq Vision fallback (age / gender) ──────────────────────────
    _gkey = (getattr(BaseConfig, 'groq_api_key', '') or os.getenv('GROQ_API_KEY', ''))
    _still_missing = (export_df['age'].isna() | export_df['gender'].isna()).any()

    if _gkey and _still_missing:
        sw.start("Groq Vision fallback (age/gender)")
        try:
            export_df = _groq_age_gender_fallback(
                export_df, input_dir, _gkey, max_workers=args.workers, rpm=25)
        except Exception as exc:
            print(f"  [Groq] age/gender fallback skipped (non-fatal): {exc}")
        sw.stop()
    elif not _gkey:
        n_miss = int((export_df['age'].isna() | export_df['gender'].isna()).sum())
        if n_miss:
            print(f"  [Groq] {n_miss} rows still missing age/gender. "
                  "Set GROQ_API_KEY to enable Vision fallback.")

    # ── Stage 5: Groq Vision fallback (voter ID) ──────────────────────────────
    _still_incomplete_vid = export_df['voterid'].apply(_is_incomplete_voter_id).any()
    if _gkey and _still_incomplete_vid:
        sw.start("Groq Vision fallback (voter ID)")
        try:
            export_df = _groq_voterid_fallback(
                export_df, input_dir, _gkey, max_workers=args.workers, rpm=25)
        except Exception as exc:
            print(f"  [Groq] voter-ID fallback skipped (non-fatal): {exc}")
        sw.stop()
    elif not _gkey:
        n_miss_vid = int(export_df['voterid'].apply(_is_incomplete_voter_id).sum())
        if n_miss_vid:
            print(f"  [Groq] {n_miss_vid} rows have empty/incomplete voter ID. "
                  "Set GROQ_API_KEY to enable Vision fallback.")

    # ── Stage 6: Azure DI catch-all fallback ──────────────────────────────────
    _di_endpoint = (getattr(BaseConfig, 'azure_di_endpoint', '')
                    or os.getenv('AZURE_DI_ENDPOINT', ''))
    _di_key      = (getattr(BaseConfig, 'azure_di_key', '')
                    or os.getenv('AZURE_DI_KEY', ''))
    _still_incomplete_any = export_df.apply(_row_is_incomplete, axis=1).any()

    if _di_endpoint and _di_key and _still_incomplete_any:
        sw.start("Azure DI catch-all fallback")
        try:
            export_df = _azure_di_catchall_fallback(
                export_df, input_dir, _di_endpoint, _di_key,
                max_workers=args.workers, rpm=15)
        except Exception as exc:
            print(f"  [Azure DI] fallback skipped (non-fatal): {exc}")
        sw.stop()

    # ── Stage 7: Post-Groq cleanup ─────────────────────────────────────────────
    sw.start("Post-cleanup (gender recovery + age sanity)")
    try:
        # Recover gender from garbled address field
        if 'gender' in export_df.columns and 'address' in export_df.columns:
            _emale_mask = (export_df['gender'].isna() &
                           export_df['address'].notna() &
                           export_df['address'].str.contains(r'emale', case=False, na=False))
            if _emale_mask.any():
                export_df.loc[_emale_mask, 'gender'] = 'Female'
                export_df.loc[_emale_mask, 'address'] = (
                    export_df.loc[_emale_mask, 'address']
                    .str.replace(r"['\s]*[Ee]male.*$", '', regex=True).str.strip())
                print(f"  Recovered {int(_emale_mask.sum())} Female values from address")

        # Clear impossible ages (< 12)
        if 'age' in export_df.columns:
            _ages_num = pd.to_numeric(export_df['age'], errors='coerce')
            _bad_age  = _ages_num.notna() & (_ages_num < 12)
            if _bad_age.any():
                export_df.loc[_bad_age, 'age'] = None
                print(f"  Cleared {int(_bad_age.sum())} impossible ages (< 12)")
    except Exception as exc:
        print(f"  Post-cleanup error (non-fatal): {exc}")
    sw.stop()

    # ── Stage 8: Caste enrichment (optional) ──────────────────────────────────
    caste_file = args.caste_file or (
        "./caste.xlsx" if os.path.isfile("./caste.xlsx") else None
    )
    if caste_file:
        sw.start("Caste / sub-caste enrichment")
        export_df = apply_caste_enrichment(export_df, caste_file)
        sw.stop()

    # ── Stage 9: Fill quality summary ─────────────────────────────────────────
    _valid_vid   = export_df['voterid'].apply(
        lambda v: bool(_VOTER_ID_STRICT_RE.match(str(v).strip())) if pd.notna(v) else False)
    _filled_age  = export_df['age'].notna() & (export_df['age'].astype(str).str.strip() != '')
    _filled_gen  = export_df['gender'].notna() & (export_df['gender'].astype(str).str.strip() != '')
    _filled_name = export_df['name'].notna() & (export_df['name'].astype(str).str.strip() != '')
    _n = len(export_df)

    print(f"\n  Final row count   : {_n:,}")
    print(f"  VoterID fill      : {int(_valid_vid.sum()):,}/{_n} "
          f"({_valid_vid.mean()*100:.1f}%)")
    print(f"  Age fill          : {int(_filled_age.sum()):,}/{_n} "
          f"({_filled_age.mean()*100:.1f}%)")
    print(f"  Gender fill       : {int(_filled_gen.sum()):,}/{_n} "
          f"({_filled_gen.mean()*100:.1f}%)")
    print(f"  Name fill         : {int(_filled_name.sum()):,}/{_n} "
          f"({_filled_name.mean()*100:.1f}%)")

    # ── Stage 10: Export to Excel ──────────────────────────────────────────────
    sw.start("Excel export")
    print("\nExporting extracted data to Excel…")
    try:
        now         = datetime.now()
        ts          = now.strftime("%d_%m_%Y_%H_%M_%S")
        # Output: ./output/Udupi/Udupi_15_06_2025_14_30_00.xlsx
        final_path  = os.path.join(constituency_out, f"{constituency}_{ts}.xlsx")
        export_df.to_excel(final_path, index=False)
        print(f"  ✅  Saved → {final_path}")
    except Exception as exc:
        print(f"  Error exporting to Excel: {exc}")
        final_path = None
    sw.stop()

    # ── Stage 11: Move completed PDFs ─────────────────────────────────────────
    if not args.keep_input:
        sw.start("Move completed PDFs")
        print("Moving processed PDFs to completed folder…")
        try:
            logic.move_completed_files(input_dir, completed_path)
            print(f"  ✅  PDFs moved → {completed_path}")
        except Exception as exc:
            print(f"  Warning — could not move files (non-fatal): {exc}")
        sw.stop()

    # ── Final summary ──────────────────────────────────────────────────────────
    print(sw.summary())
    sw.save_log(constituency_out, list_of_pdfs)

    print(f"\n{'═'*64}")
    print(f"  🎉  DONE!  Constituency: {constituency}")
    print(f"  Voters extracted : {_n:,}")
    if final_path:
        print(f"  Output Excel     : {final_path}")
    print(f"  Output folder    : {constituency_out}/")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()