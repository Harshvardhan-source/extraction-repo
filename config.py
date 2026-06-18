import os
import pandas as pd


class BaseConfig:
    # ── Local folder paths (relative; created automatically if missing) ─────────
    # When --gdrive-input is used, downloaded PDFs land in:
    #     input_dir/<constituency_name>/
    # Output Excel files are saved to:
    #     output_path/<constituency_name>/<constituency_name>_<timestamp>.xlsx
    input_dir      = r"./input"
    output_path    = r"./output"
    completed_path = r"./completed"

    # ── Poppler path (Windows only; Linux/Mac use system poppler via PATH) ──────
    # Set to None on Linux/Mac (poppler auto-detected from PATH).
    # On Windows: set to your poppler bin folder, e.g. r"C:\poppler\Library\bin"
    poppler_path = None if os.name != "nt" else r"C:\poppler\Library\bin"

    # ── Google Drive configuration ───────────────────────────────────────────────
    # gdrive_folder_id : The Drive folder that contains constituency subfolders.
    #                    Get this from the URL: drive.google.com/drive/folders/<ID>
    #                    Leave blank to skip Drive download and use local --input.
    # Example:
    #   gdrive_folder_id = "1eFf9lKcAei3IRnHnGjdQTa4eRIL8Twmn"
    gdrive_folder_id = os.getenv("GDRIVE_FOLDER_ID", "")

    # OAuth2 credentials file downloaded from Google Cloud Console
    # (APIs & Services → Credentials → OAuth 2.0 Client ID → Desktop app → Download JSON)
    credentials_file = r"credentials.json"

    # Cached token file — created automatically on first authenticated run.
    gdrive_token_file = r"token.json"

    # ── Groq API key (optional) — Vision fallback for missing age/gender ────────
    # Free key → https://console.groq.com  (14 400 req/day free tier)
    # Option A: set environment variable  GROQ_API_KEY=gsk_...
    # Option B: paste key directly below (less secure, don't commit to git):
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    # groq_api_key = "gsk_YOUR_KEY_HERE"   # ← uncomment if not using env var

    # ── Azure Document Intelligence (optional) — FINAL catch-all fallback ───────
    # Runs only on rows still incomplete after BOTH Tesseract and Groq have run.
    #   pip install azure-ai-documentintelligence
    # Option A: set environment variables AZURE_DI_ENDPOINT / AZURE_DI_KEY
    # Option B: paste values directly below (less secure):
    azure_di_endpoint = os.getenv("AZURE_DI_ENDPOINT", "")
    azure_di_key      = os.getenv("AZURE_DI_KEY", "")

    # ── Master DataFrame schema ──────────────────────────────────────────────────
    # serial_no is pre-declared so the column exists before pageSplit() runs.
    df3 = pd.DataFrame(
        data=None,
        columns=[
            "id", "page", "split",
            "polling station", "polling address",
            "voterid", "serial_no",
            "name", "father",
            "Relative Name", "Relation Type",
            "address", "age", "gender",
            "assembly_constituency_no",
            "assembly_constituency_name",
            "section_no_and_name",
            "part_no",
        ],
    )
