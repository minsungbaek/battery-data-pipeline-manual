"""
stage1_extractor.py
===================

Convert each test-item Excel summary into a unified per-cell Stage-1
dataframe with explicit input / output feature roles.

Feature spec
------------------------------------------
- Cycling   (RT 1C1C / HT 1C1C / HT 1_5C3C / RT 4C1C share schema):
    INPUT  (10): V_slippage, Cum_CE_loss, V_polarization slopes (df_dcyc 1_50, 1_endm1)
                 + delta_1_to_end SOC50 DCIR pct
                 + delta_1_to_end C10 Discharge Capacity pct
                 + delta_1_to_end Kinetic Fading
                 + ce_ex1_avg_1_25
    OUTPUT (3):  discharge_capacity_ex1 df_dcyc_1_50 / 1_endm1 / abruptness

- HT storage (per-day normalized rates):
    INPUT  (3):  2w remaining cap loss, recovered cap loss, DCIR growth
    OUTPUT (4):  4w remaining cap loss, recovered cap loss, DCIR growth, terrace thickness growth

- Rate capability (per-cell raw sheets):
    INPUT  (4):  rate_cap_ratio_3C_vs_0.5C, V_drop_3C, V_drop_5C, recovery_first_0.5C_CE
    OUTPUT (1):  rate_cap_ratio_5C_vs_0.5C

- Formation:
    INPUT  (7):  formation discharge cap (mAh/cm^2 with electrode_area input),
                 ACR, first CE (precycle C/10), formation CE (C/3),
                 DCIR0, DCIRt, DCIR_total

Output schema for every test
----------------------------
    columns = ['Barcode', 'Electrolyte', <feat_input_1>, ..., <feat_output_1>, ...]
Input and output feature lists are defined in this module.
"""

from __future__ import annotations
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

from pipeline_io import save_intermediate


# ---------------------------------------------------------------------------
# (0) xlsx-parse cache  - avoid re-parsing the same heavy xlsx on every rerun
# ---------------------------------------------------------------------------
#
# Each call stores the parsed dataframe at <xlsx_dir>/.stage1cache_<xlsx>_<sheet>.pkl
# next to the original xlsx.  On subsequent calls the cache is used as long as
# its mtime is >= the source xlsx mtime.
#

def _safe_token(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(s))[:60]


def _read_xlsx_cached(xlsx_path: str, sheet_name=None, header=None,
                      cache_tag: str = '', verbose: bool = False) -> pd.DataFrame:
    """
    Read an xlsx sheet with disk caching.  Returns a pandas DataFrame.

    Cache lives next to the xlsx as a hidden .pkl. Invalidates if xlsx mtime
    changes.  Uses pickle (always works without pyarrow) - small files anyway.
    """
    xlsx_path = Path(xlsx_path)
    sheet_tok = _safe_token(sheet_name if sheet_name is not None else 'default')
    if header is not None:
        sheet_tok += '_h' + _safe_token(str(header))
    if cache_tag:
        sheet_tok += '_' + _safe_token(cache_tag)
    cache_path = xlsx_path.parent / f".stage1cache_{xlsx_path.stem}_{sheet_tok}.pkl"

    if cache_path.exists():
        try:
            xlsx_mtime  = xlsx_path.stat().st_mtime
            cache_mtime = cache_path.stat().st_mtime
            if cache_mtime >= xlsx_mtime - 1.0:    # 1-second slack for FS noise
                if verbose:
                    print(f"[cache HIT]  {cache_path.name}")
                return pd.read_pickle(cache_path)
        except Exception:
            pass

    if verbose:
        print(f"[cache MISS] parsing {xlsx_path.name}::{sheet_name} ...")
    kwargs = {'sheet_name': sheet_name, 'engine': 'openpyxl'}
    if header is not None:
        kwargs['header'] = header
    df = pd.read_excel(xlsx_path, **kwargs)
    try:
        df.to_pickle(cache_path)
    except Exception as e:
        if verbose: print(f"[cache] save failed: {e}")
    return df


def _xlsx_excel_file(xlsx_path: str) -> pd.ExcelFile:
    """Open an ExcelFile (cheap; only reads sheet metadata)."""
    return pd.ExcelFile(xlsx_path, engine='openpyxl')


# ---------------------------------------------------------------------------
# (A) Cycling test extractor
# ---------------------------------------------------------------------------

CYCLING_INPUT_FEATURES = [
    'v_slippage_ex1_df_dcyc_1_50',
    'v_slippage_ex1_df_dcyc_1_endm1',
    'cum_ce_loss_ex1_df_dcyc_1_50',
    'cum_ce_loss_ex1_df_dcyc_1_endm1',
    'v_polarization_ex1_df_dcyc_1_50',
    'v_polarization_ex1_df_dcyc_1_endm1',
    'delta_1_to_end | SOC50 | DCIR | pct',
    'delta_1_to_end | C10 | Discharge Capacity | pct',
    'delta_1_to_end | Kinetic Fading (%)',
    'ce_ex1_avg_1_25',
]
CYCLING_OUTPUT_FEATURES = [
    # Final cycle number = TOTAL cycles run (1C + C/10 + C/3 included),
    # so it matches the corrected lifetime metric the user reports.
    # Column name 'f_end | C10 | Map Corrected Cycle' = absolute cycle index at EOL.
    'f_end | C10 | Map Corrected Cycle',
    # Slopes in '% / cycle' - invariant to cell format / electrode area
    'cycle_life_ex1_df_dcyc_1_50',
    'cycle_life_ex1_df_dcyc_1_endm1',
    'cycle_life_ex1_abruptness',
]


def extract_cycling(xlsx_path: str, raw_data_sheet: str = 'raw_data') -> pd.DataFrame:
    """
    Cycling test extractor.

    Accepts either of two xlsx formats:
      (A) Cycling processor direct output
          -> has `summary_features_ex1` + `summary_features_c10` sheets.
             We merge them on (Barcode, Electrolyte).
      (B) Notebook 2 output (`feature_outlier_removal_heatmap`)
          -> has `raw_data` sheet (already merged + outlier-flagged).

    The function auto-detects which format was passed.

    Returns dataframe with columns ['Barcode','Electrolyte', <10 input>, <3 output>].
    """
    # FAST PATH - if Notebook 1 saved a `.stage1ready.pkl` next to the xlsx,
    # skip the heavy xlsx parse entirely.
    pkl_path = Path(xlsx_path).with_suffix('').as_posix() + '.stage1ready.pkl'
    if os.path.exists(pkl_path):
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                payload = pickle.load(f)
            sx1 = payload.get('summary_features_ex1')
            sc10 = payload.get('summary_features_c10')
            if sx1 is not None and sc10 is not None:
                df = sx1.merge(sc10, on=['Barcode', 'Electrolyte'], how='outer',
                               suffixes=('', '__dup'))
                df = df[[c for c in df.columns if not c.endswith('__dup')]]
                print(f"[cycling] FAST PATH - loaded {Path(pkl_path).name} directly")
                fmt_used = "stage1ready_pkl"
            else:
                df = None
                fmt_used = None
        except Exception as e:
            print(f"[cycling] stage1ready.pkl found but unreadable ({e}); falling back to xlsx")
            df = None
            fmt_used = None
    else:
        df = None
        fmt_used = None

    if df is None:
        # Fall back to xlsx parsing
        xl = _xlsx_excel_file(xlsx_path)
        if raw_data_sheet in xl.sheet_names:
            df = _read_xlsx_cached(xlsx_path, sheet_name=raw_data_sheet)
            fmt_used = "notebook2_raw_data"
        elif 'summary_features_ex1' in xl.sheet_names and 'summary_features_c10' in xl.sheet_names:
            sx1 = _read_xlsx_cached(xlsx_path, sheet_name='summary_features_ex1')
            sc10 = _read_xlsx_cached(xlsx_path, sheet_name='summary_features_c10')
            df = sx1.merge(sc10, on=['Barcode', 'Electrolyte'], how='outer',
                           suffixes=('', '__dup'))
            df = df[[c for c in df.columns if not c.endswith('__dup')]]
            fmt_used = "notebook1_summary_ex1_plus_c10"
        else:
            raise ValueError(
                f"Could not find expected cycling sheets in {xlsx_path}.\n"
                f"  Expected EITHER 'raw_data' (notebook 2) OR\n"
                f"  'summary_features_ex1' + 'summary_features_c10' (notebook 1) OR\n"
                f"  a `.stage1ready.pkl` next to the xlsx.\n"
                f"  Available sheets: {xl.sheet_names[:10]}"
            )
        print(f"[cycling] format detected: {fmt_used}")

    # Derive missing-but-computable features from existing columns
    if 'delta_1_to_end | SOC50 | DCIR | pct' not in df.columns:
        if 'f_1 | SOC50 | DCIR' in df.columns and 'f_end | SOC50 | DCIR' in df.columns:
            f1 = pd.to_numeric(df['f_1 | SOC50 | DCIR'], errors='coerce')
            fe = pd.to_numeric(df['f_end | SOC50 | DCIR'], errors='coerce')
            df['delta_1_to_end | SOC50 | DCIR | pct'] = (fe - f1) / f1 * 100.0

    keep = ['Barcode', 'Electrolyte']
    missing = []
    for c in CYCLING_INPUT_FEATURES + CYCLING_OUTPUT_FEATURES:
        if c in df.columns:
            keep.append(c)
        else:
            missing.append(c)
            df[c] = np.nan
            keep.append(c)

    if missing:
        print(f"[cycling] WARNING - missing columns (filled with NaN): {missing}")
        if any('cum_ce_loss' in m for m in missing):
            print("[cycling] -> add `cum_ce_loss_ex1` entry to summary_1c_metrics dict in Notebook 1.")

    out = df[keep].copy()
    out = out[out['Barcode'].astype(str).str.match(r'^B\d+', na=False)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# (B) HT storage extractor - per-day normalized rates
# ---------------------------------------------------------------------------

HT_STORAGE_INPUT_FEATURES  = [
    '2w_remaining_cap_loss_per_day_pct',
    '2w_recovered_cap_loss_per_day_pct',
    '2w_DCIR_growth_per_day_pct',
    '2w_terrace_thickness_growth_per_day_pct',
    '2w_ACR_growth_per_day_pct',
]
HT_STORAGE_OUTPUT_FEATURES = [
    '4w_remaining_cap_loss_per_day_pct',
    '4w_recovered_cap_loss_per_day_pct',
    '4w_DCIR_growth_per_day_pct',
    '4w_terrace_thickness_growth_per_day_pct',
    '4w_ACR_growth_per_day_pct',
]


def _find_col(df_multi, section_substr: str, sub_substr: str):
    """Find a column where top-level header contains section_substr and
    sub-header contains sub_substr (case-insensitive)."""
    for c in df_multi.columns:
        top = str(c[0]).strip().lower()
        sub = str(c[1]).strip().lower()
        if section_substr.lower() in top and sub_substr.lower() in sub:
            return c
    return None


def extract_ht_storage(xlsx_path: str,
                       sheet_name: str = 'Summary',
                       days_2w: int = 14,
                       days_4w: int = 28,
                       thickness_xlsx_path: str | None = None) -> pd.DataFrame:
    """
    Parse HT-storage summary xlsx.  Per-day normalization of all metrics.

    Supports TWO summary formats:
      (1) Multi-level header:
            top headers '2-week', '4-week', sub-headers 'Remaining capacity%' etc.
      (2) Single-level header:
            columns like 'Remaining capacity% after 2 weeks',
            'DCIR Growth (%) after 1 month'   ('1 month' === 4 weeks here).

    `thickness_xlsx_path` (optional): if format (2) is used and the user wants
    terrace-thickness growth too, point this at the multi-level summary file
    and we'll merge that one column in.  Default None -> thickness = NaN.
    """
    # FAST PATH - if a `.stage1ready.pkl` exists next to the xlsx (e.g., made by
    # HT_Storage_Processor.ipynb), load that pre-computed summary directly.
    pkl_path = Path(xlsx_path).with_suffix('').as_posix() + '.stage1ready.pkl'
    if os.path.exists(pkl_path):
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                payload = pickle.load(f)
            ht_df = payload.get('ht_storage_summary')
            if ht_df is not None and len(ht_df) > 0:
                # Map raw HT-storage summary columns to the standard Stage-1 schema
                out = pd.DataFrame()
                bc = next((c for c in ht_df.columns if 'barcode' in c.lower()), None)
                el = next((c for c in ht_df.columns if c.lower() in ('el id', 'group', 'electrolyte')), None)
                out['Barcode']     = ht_df[bc].astype(str) if bc else ''
                out['Electrolyte'] = ht_df[el].astype(str) if el else ''

                def _g(col):
                    return pd.to_numeric(ht_df[col], errors='coerce') if col in ht_df.columns else pd.Series(np.nan, index=ht_df.index)

                # ---- Format 1 (NEW) - HT_Storage_Processor.ipynb v2: already per-day ----
                fmt_v2 = '2w_remaining_cap_loss_per_day_pct' in ht_df.columns
                if fmt_v2:
                    feat_cols = [
                        '2w_remaining_cap_loss_per_day_pct',
                        '2w_recovered_cap_loss_per_day_pct',
                        '2w_DCIR_growth_per_day_pct',
                        '2w_terrace_thickness_growth_per_day_pct',
                        '2w_ACR_growth_per_day_pct',
                        '4w_remaining_cap_loss_per_day_pct',
                        '4w_recovered_cap_loss_per_day_pct',
                        '4w_DCIR_growth_per_day_pct',
                        '4w_terrace_thickness_growth_per_day_pct',
                        '4w_ACR_growth_per_day_pct',
                    ]
                    for c in feat_cols:
                        out[c] = _g(c)
                else:
                    # ---- Format 0/1 (raw retention % columns from picker-style notebook) ----
                    rem2 = _g('Remaining capacity% after 2 weeks')
                    rec2 = _g('Recovered capacity% after 2 weeks')
                    dci2 = _g('DCIR Growth (%) after 2 weeks')
                    rem4 = _g('Remaining capacity% after 1 month')
                    rec4 = _g('Recovered capacity% after 1 month')
                    dci4 = _g('DCIR Growth (%) after 1 month')

                    # thickness - match either "after 2 weeks/1 month" (HT_Storage_Processor v3)
                    # or "4-week/4 week" (older multi-level summary)
                    def _find(substrs_all):
                        for col in ht_df.columns:
                            cl = col.lower()
                            if all(s in cl for s in substrs_all):
                                return col
                        return None
                    th2_col = _find(['thickness','growth','2 weeks']) or _find(['thickness','growth','2-week'])
                    th4_col = _find(['thickness','growth','1 month']) or _find(['thickness','growth','4-week']) or _find(['thickness','growth','4 week'])
                    acr2_col = _find(['acr','growth','2 weeks'])
                    acr4_col = _find(['acr','growth','1 month']) or _find(['acr','growth','4-week']) or _find(['acr','growth','4 week'])

                    th2  = _g(th2_col)  if th2_col  else pd.Series(np.nan, index=ht_df.index)
                    th4  = _g(th4_col)  if th4_col  else pd.Series(np.nan, index=ht_df.index)
                    acr2 = _g(acr2_col) if acr2_col else pd.Series(np.nan, index=ht_df.index)
                    acr4 = _g(acr4_col) if acr4_col else pd.Series(np.nan, index=ht_df.index)

                    out['2w_remaining_cap_loss_per_day_pct']        = (100.0 - rem2) / days_2w
                    out['2w_recovered_cap_loss_per_day_pct']        = (100.0 - rec2) / days_2w
                    out['2w_DCIR_growth_per_day_pct']               = dci2 / days_2w
                    out['4w_remaining_cap_loss_per_day_pct']        = (100.0 - rem4) / days_4w
                    out['4w_recovered_cap_loss_per_day_pct']        = (100.0 - rec4) / days_4w
                    out['4w_DCIR_growth_per_day_pct']               = dci4 / days_4w
                    out['2w_terrace_thickness_growth_per_day_pct']  = th2  / days_2w
                    out['4w_terrace_thickness_growth_per_day_pct']  = th4  / days_4w
                    out['2w_ACR_growth_per_day_pct']                = acr2 / days_2w
                    out['4w_ACR_growth_per_day_pct']                = acr4 / days_4w

                out = out[out['Barcode'].astype(str).str.match(r'^B\d+', na=False)].reset_index(drop=True)
                print(f"[ht_storage] FAST PATH - loaded {Path(pkl_path).name} ({'v2 pre-computed' if fmt_v2 else 'v1 raw'})")
                return out
        except Exception as e:
            print(f"[ht_storage] stage1ready.pkl unreadable ({e}); falling back to xlsx")

    # Detect format by trying multi-level first
    df_multi = None
    try:
        df_multi = _read_xlsx_cached(xlsx_path, sheet_name=sheet_name, header=[0, 1])
        # multi-level if at least one top header looks like '2-week' or '4-week'
        tops = {str(c[0]).strip().lower() for c in df_multi.columns}
        is_multi = any(t in tops for t in ('2-week', '4-week'))
    except Exception:
        is_multi = False

    if is_multi:
        df = df_multi
        bc_col = _find_col(df, '', 'Barcode')
        el_col = _find_col(df, '', 'EL ID')
        if bc_col is None: bc_col = df.columns[0]
        if el_col is None: el_col = df.columns[1]
        out = pd.DataFrame({
            'Barcode':     df[bc_col].astype(str),
            'Electrolyte': df[el_col].astype(str),
        })
        def get_num(section, sub):
            col = _find_col(df, section, sub)
            return pd.to_numeric(df[col], errors='coerce') if col is not None else pd.Series(np.nan, index=df.index)
        rem2 = get_num('2-week', 'Remaining capacity%')
        rec2 = get_num('2-week', 'Recovered capacity%')
        dci2 = get_num('2-week', 'DCIR growth')
        rem4 = get_num('4-week', 'Remaining capacity%')
        rec4 = get_num('4-week', 'Recovered capacity%')
        dci4 = get_num('4-week', 'DCIR growth')
        th4  = get_num('4-week', 'Cell terrace thickness growth')
        format_used = 'multi_level'
    else:
        # Single-level format
        df = _read_xlsx_cached(xlsx_path, sheet_name=sheet_name)
        df.columns = [str(c).strip() for c in df.columns]
        # Find Barcode / EL ID
        bc_col = next((c for c in df.columns if 'barcode' in c.lower()), df.columns[0])
        el_col = next((c for c in df.columns if c.lower() in ('el id', 'electrolyte', 'group')), df.columns[1])
        out = pd.DataFrame({
            'Barcode':     df[bc_col].astype(str),
            'Electrolyte': df[el_col].astype(str),
        })
        def get_one(*candidates):
            for cand in candidates:
                for c in df.columns:
                    if cand.lower() in c.lower():
                        return pd.to_numeric(df[c], errors='coerce')
            return pd.Series(np.nan, index=df.index)
        rem2 = get_one('Remaining capacity% after 2 weeks', 'Remaining capacity% after 2week')
        rec2 = get_one('Recovered capacity% after 2 weeks', 'Recovered capacity% after 2week')
        dci2 = get_one('DCIR Growth (%) after 2 weeks', 'DCIR growth (%) after 2week')
        rem4 = get_one('Remaining capacity% after 1 month', 'Remaining capacity% after 4 weeks',
                       'Remaining capacity% after 1month')
        rec4 = get_one('Recovered capacity% after 1 month', 'Recovered capacity% after 4 weeks',
                       'Recovered capacity% after 1month')
        dci4 = get_one('DCIR Growth (%) after 1 month', 'DCIR Growth (%) after 4 weeks',
                       'DCIR Growth (%) after 1month')
        th4  = pd.Series(np.nan, index=df.index)   # not in single-level format
        # Optionally merge thickness from a second xlsx
        if thickness_xlsx_path is not None:
            try:
                df_th = _read_xlsx_cached(thickness_xlsx_path, sheet_name=sheet_name, header=[0,1])
                bc_th = _find_col(df_th, '', 'Barcode') or df_th.columns[0]
                col_th = _find_col(df_th, '4-week', 'Cell terrace thickness growth')
                if col_th is not None:
                    th_map = pd.Series(
                        pd.to_numeric(df_th[col_th], errors='coerce').values,
                        index=df_th[bc_th].astype(str)
                    ).to_dict()
                    th4 = out['Barcode'].map(th_map)
            except Exception as e:
                print(f"[ht_storage] could not merge thickness from {thickness_xlsx_path}: {e}")
        format_used = 'single_level'

    out['2w_remaining_cap_loss_per_day_pct'] = (100.0 - rem2) / days_2w
    out['2w_recovered_cap_loss_per_day_pct'] = (100.0 - rec2) / days_2w
    out['2w_DCIR_growth_per_day_pct']         = dci2 / days_2w
    out['4w_remaining_cap_loss_per_day_pct'] = (100.0 - rem4) / days_4w
    out['4w_recovered_cap_loss_per_day_pct'] = (100.0 - rec4) / days_4w
    out['4w_DCIR_growth_per_day_pct']         = dci4 / days_4w
    out['4w_terrace_thickness_growth_per_day_pct'] = th4 / days_4w
    # New optional features - fill NaN if not present in this xlsx format
    if '2w_terrace_thickness_growth_per_day_pct' not in out.columns:
        out['2w_terrace_thickness_growth_per_day_pct'] = np.nan
    if '2w_ACR_growth_per_day_pct' not in out.columns:
        out['2w_ACR_growth_per_day_pct'] = np.nan
    if '4w_ACR_growth_per_day_pct' not in out.columns:
        out['4w_ACR_growth_per_day_pct'] = np.nan

    out = out[out['Barcode'].astype(str).str.match(r'^B\d+', na=False)].reset_index(drop=True)
    print(f"[ht_storage] format={format_used}")
    return out


# ---------------------------------------------------------------------------
# (C) Rate capability extractor
# ---------------------------------------------------------------------------

RATE_CAP_INPUT_FEATURES  = [
    'rate_cap_ratio_3C_vs_0.5C',
    'rate_cap_voltage_drop_3C_vs_0.5C_V',
    'rate_cap_voltage_drop_5C_vs_0.5C_V',
    # Trapped Li fraction during 5C: of the recovery 0.5C discharge,
    # how much came from previously-trapped Li (not from this cycle's charge).
    # = 1 - Q_chg_recovery / Q_disch_recovery
    'rate_cap_trapped_Li_fraction_after_5C',
]
RATE_CAP_OUTPUT_FEATURES = [
    'rate_cap_ratio_5C_vs_0.5C',
]

# Default protocol observation: each rate has 5 consecutive cycles, last cycle of
# each block is the equilibrated value.  Cycle 1 is formation/0.5C cycle with
# Si activation artifact.  Cycle 31 is the first 0.5C recovery cycle after 5C.
DEFAULT_RATE_LAST_CYCLE = {
    '0.5C': 5,    # last 0.5C cycle
    '1C':   10,
    '2C':   15,
    '3C':   20,
    '4C':   25,
    '5C':   30,
}
DEFAULT_RECOVERY_CYCLE = 31  # first 0.5C cycle after 5C block - trapped-Li signal


def extract_rate_cap(xlsx_path: str,
                     cell_codes_sheet: str = 'cell_codes',
                     rate_last_cycle: dict | None = None,
                     recovery_cycle: int = DEFAULT_RECOVERY_CYCLE) -> pd.DataFrame:
    """
    Per-cell rate capability extractor.

    FAST PATH: if a `.stage1ready.pkl` exists next to the xlsx (created by
    `Rate_Capability_Processor.ipynb`), load it directly - skips heavy parsing.

    Otherwise: reads the per-cell sheets in the source workbook
    file and computes the 5 features using fixed cycle indices.
    """
    # FAST PATH
    pkl_path = Path(xlsx_path).with_suffix('').as_posix() + '.stage1ready.pkl'
    if os.path.exists(pkl_path):
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                payload = pickle.load(f)
            df = payload.get('rate_cap_summary')
            if df is not None and 'Barcode' in df.columns and 'Electrolyte' in df.columns:
                print(f"[rate_cap] FAST PATH - loaded {Path(pkl_path).name} directly")
                return df.copy()
        except Exception as e:
            print(f"[rate_cap] stage1ready.pkl unreadable ({e}); falling back to xlsx")
    if rate_last_cycle is None:
        rate_last_cycle = DEFAULT_RATE_LAST_CYCLE

    xl = _xlsx_excel_file(xlsx_path)
    if cell_codes_sheet not in xl.sheet_names:
        # Graceful skip: rate_cap source xlsx is missing per-cell sheets.
        # Common cause: user pointed at the rate_cap_output.xlsx (Processor's
        # OUTPUT) which lacks cell_codes/per-cell sheets. Return an empty df
        # so Stage 2 can continue without rate_cap features.
        print(f"[rate_cap] WARNING - '{cell_codes_sheet}' sheet missing in "
              f"{Path(xlsx_path).name}.\n            "
              f"Rate-cap features will be SKIPPED for this DOE.\n            "
              f"To recover them: regenerate the .stage1ready.pkl by running "
              f"`Rate_Capability_Processor.ipynb`, OR point the pipeline at "
              f"the original per-cell rate-capability source file.")
        return pd.DataFrame(columns=['Barcode', 'Electrolyte']
                            + RATE_CAP_INPUT_FEATURES + RATE_CAP_OUTPUT_FEATURES)

    cells = _read_xlsx_cached(xlsx_path, sheet_name=cell_codes_sheet)
    cells.columns = [str(c).strip() for c in cells.columns]

    rows = []
    for _, r in cells.iterrows():
        bc = str(r.get('Barcode')).strip()
        el = str(r.get('Group')).strip()
        if not re.match(r'^B\d+', bc):
            continue
        if bc not in xl.sheet_names:
            continue

        cell_df = _read_xlsx_cached(xlsx_path, sheet_name=bc)
        cell_df['Cycle'] = pd.to_numeric(cell_df.get('Cycle'), errors='coerce')

        def _val(cycle, col):
            sub = cell_df[cell_df['Cycle'] == cycle]
            if sub.empty: return np.nan
            v = pd.to_numeric(sub[col], errors='coerce').iloc[0]
            return float(v) if pd.notna(v) else np.nan

        # Discharge capacities at end of each rate block
        q_05c = _val(rate_last_cycle['0.5C'], 'Discharge Capacity')
        q_3c  = _val(rate_last_cycle['3C'],   'Discharge Capacity')
        q_5c  = _val(rate_last_cycle['5C'],   'Discharge Capacity')

        # Average discharge voltages
        v_05c = _val(rate_last_cycle['0.5C'], 'Average Discharge Voltage')
        v_3c  = _val(rate_last_cycle['3C'],   'Average Discharge Voltage')
        v_5c  = _val(rate_last_cycle['5C'],   'Average Discharge Voltage')

        # Trapped Li fraction during the first recovery cycle after the 5C block.
        q_recov_disch = _val(recovery_cycle, 'Discharge Capacity')
        q_recov_chg = _val(recovery_cycle, 'Charge Capacity')
        trapped_li = (1.0 - q_recov_chg / q_recov_disch) if (
            pd.notna(q_recov_chg) and pd.notna(q_recov_disch) and q_recov_disch > 0
        ) else np.nan

        ratio_3 = q_3c / q_05c if (pd.notna(q_05c) and q_05c != 0) else np.nan
        ratio_5 = q_5c / q_05c if (pd.notna(q_05c) and q_05c != 0) else np.nan

        rows.append({
            'Barcode': bc,
            'Electrolyte': el,
            'rate_cap_ratio_3C_vs_0.5C':            ratio_3,
            'rate_cap_voltage_drop_3C_vs_0.5C_V':   (v_3c - v_05c) if pd.notna(v_3c) and pd.notna(v_05c) else np.nan,
            'rate_cap_voltage_drop_5C_vs_0.5C_V':   (v_5c - v_05c) if pd.notna(v_5c) and pd.notna(v_05c) else np.nan,
            'rate_cap_trapped_Li_fraction_after_5C': trapped_li,
            'rate_cap_ratio_5C_vs_0.5C':            ratio_5,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (D) Formation extractor
# ---------------------------------------------------------------------------

FORMATION_FEATURES = [
    'formation_discharge_cap_areal_mAh_per_cm2',
    'formation_first_CE_pct_precycle_C10',
    'formation_CE_pct_C3',
    'formation_DCIR0_mOhm',
    'formation_DCIRt_mOhm',
    'formation_DCIR_total_mOhm',
]


def extract_formation(xlsx_path: str,
                      electrode_area_cm2: float,
                      sheet_name: str = 'Summary',
                      barcode_filter_regex: str | None = None) -> pd.DataFrame:
    """
    Formation summary extractor.

    `electrode_area_cm2` must be supplied by the user (input box in the Stage-2
    notebook) so that the cap-check discharge capacity is converted to mAh/cm^2.

    `barcode_filter_regex` (optional): if the formation summary contains cells
    from multiple DOEs, provide a regex to select just this DOE's cells.

    Column names in this version of the summary
    -------------------------------------------
        Barcode, EL ID,
        PRE_c1_CE (%)               -> formation_first_CE_pct_precycle_C10
        CAP_c1_CE (%)               -> formation_CE_pct_C3
        CAP_c1_dchg_cap (Ah)        -> areal cap (- 1000 / area_cm^2)
        dcir0 (mOhms)               -> formation_DCIR0_mOhm
        dcirt (mOhms)               -> formation_DCIRt_mOhm
        dcir  (mOhms)               -> formation_DCIR_total_mOhm
        (no explicit ACR column -> set NaN, optionally use 'dcir0 1C' as proxy)
    """
    df = _read_xlsx_cached(xlsx_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    def _pick(*candidates):
        for cand in candidates:
            for c in df.columns:
                if c.lower().strip() == cand.lower().strip():
                    return c
        for cand in candidates:
            for c in df.columns:
                if cand.lower() in c.lower():
                    return c
        return None

    bc_col   = _pick('Barcode')
    el_col   = _pick('EL ID', 'Group', 'Electrolyte')
    cap_col  = _pick('CAP_c1_dchg_cap (Ah)', 'CAP_c1_dchg_cap',
                     'cap_check_dchg', 'Discharge Capacity (Ah)')
    fce_col  = _pick('PRE_c1_CE (%)', 'PRE_c1_CE', 'precycle_C1_CE', 'first CE (%)')
    fmce_col = _pick('CAP_c1_CE (%)', 'CAP_c1_CE', 'formation_CE')
    dcir0    = _pick('dcir0 (mOhms)', 'dcir0 (mOhm)', 'DCIR0', 'dcir0')
    dcirt    = _pick('dcirt (mOhms)', 'dcirt (mOhm)', 'DCIRt', 'dcirt')
    dcirtot  = _pick('dcir (mOhms)', 'dcir (mOhm)', 'DCIR', 'dcir total')

    out = pd.DataFrame({
        'Barcode':     df[bc_col].astype(str)  if bc_col else pd.Series(dtype=str),
        'Electrolyte': df[el_col].astype(str)  if el_col else pd.Series(dtype=str),
    })

    if cap_col is not None:
        cap_Ah = pd.to_numeric(df[cap_col], errors='coerce')
        out['formation_discharge_cap_areal_mAh_per_cm2'] = (cap_Ah * 1000.0) / float(electrode_area_cm2)
    else:
        out['formation_discharge_cap_areal_mAh_per_cm2'] = np.nan

    out['formation_first_CE_pct_precycle_C10'] = pd.to_numeric(df[fce_col],   errors='coerce') if fce_col   else np.nan
    out['formation_CE_pct_C3']                 = pd.to_numeric(df[fmce_col],  errors='coerce') if fmce_col  else np.nan
    out['formation_DCIR0_mOhm']                = pd.to_numeric(df[dcir0],     errors='coerce') if dcir0     else np.nan
    out['formation_DCIRt_mOhm']                = pd.to_numeric(df[dcirt],     errors='coerce') if dcirt     else np.nan
    out['formation_DCIR_total_mOhm']           = pd.to_numeric(df[dcirtot],   errors='coerce') if dcirtot   else np.nan

    # Drop rows where barcode is not a valid B... pattern
    out = out[out['Barcode'].astype(str).str.match(r'^B\d+', na=False)].reset_index(drop=True)

    # Apply DOE-specific filter if provided
    if barcode_filter_regex:
        out = out[out['Barcode'].str.match(barcode_filter_regex, na=False)].reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# (E) Top-level driver
# ---------------------------------------------------------------------------


def run_stage1(test_type: str, xlsx_path: str, output_dir: str,
               electrode_area_cm2=None,
               verbose: bool = True, **kwargs) -> str:
    test_type = test_type.strip().lower()
    from pathlib import Path as _Path
    output_dir = _Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    if test_type in ('cycling', 'cycle'):
        df = extract_cycling(xlsx_path, **{k: v for k, v in kwargs.items()
                                            if k in ('raw_data_sheet',)})
    elif test_type in ('ht_storage', 'storage', 'ht-storage'):
        df = extract_ht_storage(xlsx_path, **{k: v for k, v in kwargs.items()
                                                if k in ('sheet_name', 'days_2w', 'days_4w',
                                                         'thickness_xlsx_path')})
    elif test_type in ('rate_cap', 'rate', 'rate_capability'):
        df = extract_rate_cap(xlsx_path, **{k: v for k, v in kwargs.items()
                                              if k in ('cell_codes_sheet', 'rate_last_cycle', 'recovery_cycle')})
    elif test_type == 'formation':
        if electrode_area_cm2 is None:
            raise ValueError("formation test requires electrode_area_cm2")
        df = extract_formation(xlsx_path, electrode_area_cm2=electrode_area_cm2,
                               **{k: v for k, v in kwargs.items()
                                  if k in ('sheet_name', 'barcode_filter_regex')})
    else:
        raise ValueError(f"Unknown test_type: {test_type!r}")

    if df.empty:
        if verbose: print(f"[stage1:{test_type}] no rows extracted"); return ''

    out_path = save_intermediate(df, output_dir, 'summary_features')

    import json as _json
    meta = {
        'test_type':          test_type,
        'source_xlsx':        str(_Path(xlsx_path).resolve()),
        'n_cells':            int(len(df)),
        'n_features':         int(df.shape[1] - 2),
        'electrode_area_cm2': electrode_area_cm2,
        'days_2w':            kwargs.get('days_2w'),
        'days_4w':            kwargs.get('days_4w'),
    }
    with open(output_dir / '_stage1_meta.json', 'w', encoding='utf-8') as f:
        _json.dump(meta, f, indent=2)

    if verbose:
        print(f"[stage1:{test_type}] {len(df)} cells x {df.shape[1]-2} features  ->  {out_path}")

    return out_path


def get_input_output_features(test_type: str) -> tuple:
    """Return (input_features, output_features) for one test type."""
    t = test_type.strip().lower()
    if t in ('cycling', 'cycle'):
        return CYCLING_INPUT_FEATURES, CYCLING_OUTPUT_FEATURES
    if t in ('ht_storage', 'storage', 'ht-storage'):
        return HT_STORAGE_INPUT_FEATURES, HT_STORAGE_OUTPUT_FEATURES
    if t in ('rate_cap', 'rate', 'rate_capability'):
        return RATE_CAP_INPUT_FEATURES, RATE_CAP_OUTPUT_FEATURES
    if t == 'formation':
        return FORMATION_FEATURES, []
    return [], []

