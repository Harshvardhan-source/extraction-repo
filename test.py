#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caste / Sub-Caste Enrichment  —  test.py
=========================================
Reads an already-extracted voter Excel file, applies sub-caste detection
(name-matching against the caste.xlsx sheets) and top-level caste lookup,
then writes the enriched result.

Usage (standalone):
  python test.py --input  ./output/Udupi/Udupi_15_06_2025.xlsx \\
                 --caste  ./caste.xlsx \\
                 --output ./output/Udupi/Udupi_with_caste.xlsx

Or import as a module from main.py:
  from test import caste_function, sub_caste_function
"""

import argparse
import sys
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS  (importable by main.py)
# ═══════════════════════════════════════════════════════════════════════════════

def caste_function(df3: pd.DataFrame, caste_file: str) -> pd.DataFrame:
    """
    Merge the voter DataFrame with the 'Caste' sheet from caste_file.

    The 'Caste' sheet must have exactly two columns:
        Col A  →  Sub_Caste  (internal caste name used in sub_caste column)
        Col B  →  caste      (display caste name added to the output)

    A left-join on sub_caste → Sub_Caste adds the 'caste' column.
    Rows with no sub_caste match get NaN in 'caste'.
    """
    df = df3.copy(deep=True)
    caste_df = pd.read_excel(caste_file, sheet_name='Caste')
    caste_df.columns = ["Sub_Caste", "caste"]
    new_df = pd.merge(df, caste_df, how='left',
                      left_on='sub_caste', right_on='Sub_Caste')
    # Drop the duplicate join-key column
    if "Sub_Caste" in new_df.columns:
        new_df.drop(columns=["Sub_Caste"], inplace=True)
    return new_df


def sub_caste_function(df3: pd.DataFrame, caste_file: str) -> pd.DataFrame:
    """
    Tag each voter row with a 'sub_caste' value by matching the voter name
    and father name against the name lists in each non-'Caste' sheet of
    caste_file.

    Sheet structure expected (one community per sheet):
        Sheet name  →  the sub_caste label (e.g. "Bunt", "GSB", "Billava")
        Column "Names"  →  list of name tokens to match (case-insensitive)

    Matching order:
        1. Father name  (higher priority — community names appear more often)
        2. Voter name

    Bug fixes vs the original test.py:
        • rowcount was inside the inner loop → incremented N×per row (fixed)
        • condition was `if not name or pd.isnull(name)` — INVERTED (fixed)
        • exit(0) on file error removed — non-fatal now, returns original df
    """
    try:
        df = df3.copy(deep=True)
        df["name"] = df["name"].fillna("not available")
        df["sub_caste"] = df.get("sub_caste", None)   # preserve if already set

        df_name = pd.ExcelFile(caste_file)

        for sheet in df_name.sheet_names:
            if sheet.lower() == "caste":
                continue

            caste_df = pd.read_excel(caste_file, sheet_name=sheet)
            if caste_df.empty or "Names" not in caste_df.columns:
                continue

            caste_list = [str(x).lower() for x in caste_df["Names"] if pd.notna(x)]
            if not caste_list:
                continue

            # ── Match against father name ─────────────────────────────────────
            for rowcount, name in enumerate(df["father"]):
                if rowcount >= len(df):
                    break
                for token in caste_list:
                    if token and token in str(name).lower():
                        df.at[rowcount, "sub_caste"] = sheet
                        break   # first match wins per row

            # ── Match against voter name ──────────────────────────────────────
            for rowcount, name in enumerate(df["name"]):
                if rowcount >= len(df):
                    break
                if name and not pd.isnull(name):
                    for token in caste_list:
                        if token and token in str(name).lower():
                            df.at[rowcount, "sub_caste"] = sheet
                            break

    except Exception as exc:
        print(f"Error while reading caste file: {exc}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(
        description="Add sub_caste / caste columns to an extracted voter Excel file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python test.py \\
      --input  ./output/Udupi/Udupi_15_06_2025_14_30_00.xlsx \\
      --caste  ./caste.xlsx \\
      --output ./output/Udupi/Udupi_with_caste.xlsx
        """
    )
    ap.add_argument(
        "--input", "-i", required=True, metavar="XLSX",
        help="Path to the voter Excel file produced by main.py"
    )
    ap.add_argument(
        "--caste", "-c", default="./caste.xlsx", metavar="XLSX",
        help="Path to caste.xlsx (default: ./caste.xlsx)"
    )
    ap.add_argument(
        "--output", "-o", default=None, metavar="XLSX",
        help="Output path. Defaults to <input>_with_caste.xlsx"
    )
    ap.add_argument(
        "--sheet", default="Extract", metavar="NAME",
        help="Sheet name to read from the input Excel (default: Extract). "
             "Use 'Sheet1' or '0' (for first sheet) if needed."
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Read input Excel ──────────────────────────────────────────────────────
    print(f"Reading input file: {args.input}")
    try:
        sheet = args.sheet
        if sheet.isdigit():
            sheet = int(sheet)
        df3 = pd.read_excel(args.input, sheet_name=sheet)
    except Exception as exc:
        # If the named sheet doesn't exist, try the first sheet
        try:
            print(f"  Sheet '{args.sheet}' not found, trying first sheet…")
            df3 = pd.read_excel(args.input, sheet_name=0)
        except Exception as exc2:
            sys.exit(f"❌  Could not read input file: {exc2}")

    print(f"  Rows loaded: {len(df3):,}")

    # ── Ensure required columns exist ─────────────────────────────────────────
    for col in ("name", "father"):
        if col not in df3.columns:
            df3[col] = ""

    # ── Sub-caste detection ───────────────────────────────────────────────────
    print(f"\nRunning sub-caste detection from: {args.caste}")
    df_with_subcaste = sub_caste_function(df3, args.caste)

    # ── Top-level caste lookup ────────────────────────────────────────────────
    print("Running top-level caste lookup…")
    final_df = caste_function(df_with_subcaste, args.caste)

    print(f"\nSub-caste tagged : {final_df['sub_caste'].notna().sum():,} rows")
    if 'caste' in final_df.columns:
        print(f"Caste tagged     : {final_df['caste'].notna().sum():,} rows")

    # ── Select export columns ─────────────────────────────────────────────────
    # Include caste/sub_caste; preserve all original columns
    print(f"\nColumn summary:\n  {list(final_df.columns)}")

    # ── Save output ───────────────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        base = args.input.rsplit('.', 1)[0]
        out_path = base + "_with_caste.xlsx"

    print(f"\nSaving enriched file → {out_path}")
    final_df.to_excel(out_path, index=False)
    print(f"  ✅  Done  ({len(final_df):,} rows)")


if __name__ == "__main__":
    main()
