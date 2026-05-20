"""
stage2_aggregator.py
====================

Stage 2 -- per-DOE aggregation across multiple test items.

Workflow
--------
    Stage 1 outputs (per test item, with _stage1_meta.json)
       |
    Stage 2 (this module):
       1. Auto-discover stage1_<test_item>/ subfolders
       2. For each, load summary_features.{parquet|pkl} and read its test_type
          from _stage1_meta.json
       3. Tag each feature column with its test_item + role (INPUT/OUTPUT)
       4. Filter formation cells to the union of cycling-test barcodes
          (formation summary often pools many DOEs' cells)
       5. Outer-merge all per-test summaries on (Barcode, Electrolyte)
       6. Fit/load FeatureScaler, detect & remove bad cells
       7. Lot-level mean within (electrolyte)
       8. Compute delta vs baseline electrolyte: % (human) AND sigma (ML)
       9. Output xlsx (with input/output sheets separated) + parquet
"""

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from feature_transformations import FeatureScaler, flag_bad_cells
from stage1_extractor import get_input_output_features
from pipeline_io import load_intermediate, save_intermediate


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------

def discover_stage1_folders(doe_root: str, prefix: str = 'stage1_') -> dict:
    """Return {test_item_name: stage1_path} for all stage1_* subfolders."""
    doe_root = Path(doe_root)
    out = {}
    for p in sorted(doe_root.iterdir()):
        if p.is_dir() and p.name.startswith(prefix):
            out[p.name[len(prefix):]] = str(p)
    return out


def _load_stage1(stage1_dir: str) -> tuple:
    """Load (df, meta) from one stage1 folder."""
    p = Path(stage1_dir)
    df = load_intermediate(p, 'summary_features')
    meta_path = p / '_stage1_meta.json'
    meta = {}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    return df, meta


# ---------------------------------------------------------------------------
# Tagging features with test_item + role
# ---------------------------------------------------------------------------

def _tag_columns(df: pd.DataFrame, test_item: str, test_type: str,
                 id_cols=('Barcode', 'Electrolyte')) -> tuple:
    inputs, outputs = get_input_output_features(test_type)
    role_lookup = {f: 'INPUT' for f in inputs}
    role_lookup.update({f: 'OUTPUT' for f in outputs})

    rename = {}
    role_map = {}
    for c in df.columns:
        if c in id_cols:
            continue
        new = f"{test_item} | {c}"
        rename[c] = new
        role_map[new] = role_lookup.get(c, 'INPUT')

    return df.rename(columns=rename), role_map


# ---------------------------------------------------------------------------
# Lot-average + baseline-delta helpers
# ---------------------------------------------------------------------------

def aggregate_cells_to_lots(df: pd.DataFrame,
                             group_col: str = 'Electrolyte',
                             id_col: str = 'Barcode') -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c not in (id_col, group_col)
                 and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(group_col, dropna=False)
    agg = grouped[feat_cols].mean().reset_index()
    agg['n_cells_used'] = grouped.size().reindex(agg[group_col]).values
    return agg


def aggregate_cells_to_lots_std(df: pd.DataFrame,
                                 group_col: str = 'Electrolyte',
                                 id_col: str = 'Barcode') -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c not in (id_col, group_col)
                 and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(group_col, dropna=False)
    std_df = grouped[feat_cols].std(ddof=1).reset_index()
    std_df['n_cells_used'] = grouped.size().reindex(std_df[group_col]).values
    return std_df


def compute_delta_vs_baseline(lot_df: pd.DataFrame,
                              baseline_id: str,
                              group_col: str = 'Electrolyte',
                              mode: str = 'subtract') -> pd.DataFrame:
    if baseline_id not in lot_df[group_col].astype(str).values:
        raise ValueError(f"Baseline {baseline_id!r} not in {group_col}.")

    base_row = lot_df[lot_df[group_col].astype(str) == baseline_id].iloc[0]
    feat_cols = [c for c in lot_df.columns if c not in (group_col, 'n_cells_used')
                 and pd.api.types.is_numeric_dtype(lot_df[c])]

    rows = []
    for _, row in lot_df.iterrows():
        is_baseline = (str(row[group_col]) == baseline_id)
        d = {group_col: row[group_col]}
        if 'n_cells_used' in lot_df.columns:
            d['n_cells_used'] = row['n_cells_used']
        d['_is_baseline'] = is_baseline
        for c in feat_cols:
            v, b = row[c], base_row[c]
            if is_baseline:
                d[c] = 0.0 if pd.notna(b) else np.nan
                continue
            if pd.isna(v) or pd.isna(b):
                d[c] = np.nan
                continue
            if mode == 'subtract':
                d[c] = v - b
            elif mode == 'pct':
                d[c] = ((v - b) / abs(b) * 100.0) if b != 0 else np.nan
            else:
                d[c] = np.nan
        rows.append(d)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['_is_baseline', group_col],
                               ascending=[False, True]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Color-coded headline summary
# ---------------------------------------------------------------------------

DAYS_4W = 28.0

HEADLINE_FEATURES_DEFAULT = [
    ('5C retention',        'rate_cap_ratio_5C_vs_0.5C', +1, None,
        lambda x: 100.0 * x),
    ('Cycle # (RT 1C1C)',   'RT_1C1C',  +1, 'Map Corrected Cycle', None),
    ('Cycle # (HT 1C1C)',   'HT_1C1C',  +1, 'Map Corrected Cycle', None),
    ('Cycle # (HT 1.5C3C)', 'HT_1_5C',  +1, 'Map Corrected Cycle', None),
    ('Remaining (4w)',      '4w_remaining_cap_loss_per_day_pct', +1, None,
        lambda x: 100.0 - DAYS_4W * x),
    ('Recovery (4w)',       '4w_recovered_cap_loss_per_day_pct', +1, None,
        lambda x: 100.0 - DAYS_4W * x),
    ('DC-IR growth (4w)',   '4w_DCIR_growth_per_day_pct',        -1, None,
        lambda x: DAYS_4W * x),
    ('Terrace thick (4w)',  '4w_terrace_thickness_growth_per_day_pct', -1, None,
        lambda x: DAYS_4W * x),
    ('ACR growth (4w)',     '4w_ACR_growth_per_day_pct',         -1, None,
        lambda x: DAYS_4W * x),
    ('Formation 1st CE',    'formation_first_CE_pct_precycle_C10', +1, None, None),
]

HEADLINE_DISPLAY_MULTIPLIERS = {
    'rate_cap_ratio': 100.0,
}


def _display_multiplier(col_name: str) -> float:
    s = str(col_name).lower()
    for k, m in HEADLINE_DISPLAY_MULTIPLIERS.items():
        if k in s:
            return m
    return 1.0


def _match_headline_column(columns, pattern, *, must_contain=None):
    cands = [c for c in columns if pattern in c]
    if must_contain is not None:
        cands = [c for c in cands if must_contain in c]
    return cands[0] if cands else None


def build_color_coded_summary(lot_df: pd.DataFrame,
                              pct_delta: pd.DataFrame,
                              baseline_id: str,
                              headline_features=None,
                              group_col: str = 'Electrolyte') -> tuple:
    if headline_features is None:
        headline_features = HEADLINE_FEATURES_DEFAULT

    resolved = []
    for spec in headline_features:
        if len(spec) == 3:
            disp, pat, direction = spec; must = None; transform = None
        elif len(spec) == 4:
            disp, pat, direction, must = spec; transform = None
        elif len(spec) == 5:
            disp, pat, direction, must, transform = spec
        else:
            continue
        col = _match_headline_column(lot_df.columns, pat, must_contain=must)
        if col is None:
            continue
        resolved.append((disp, col, direction, transform))

    def _apply(t, x):
        if t is None or pd.isna(x): return x
        try: return float(t(x))
        except Exception: return np.nan

    rows = []
    for _, lot_row in pct_delta.iterrows():
        el = str(lot_row[group_col])
        match = lot_df[lot_df[group_col].astype(str) == el]
        if match.empty:
            continue
        lot_data = match.iloc[0]
        n_cells = lot_data.get('n_cells_used', np.nan)
        out = {group_col: el, 'n': int(n_cells) if pd.notna(n_cells) else 0}
        for disp, col, direction, transform in resolved:
            raw_v = lot_data.get(col, np.nan)
            bv_raw = lot_df.loc[lot_df[group_col].astype(str) == baseline_id, col]
            base_raw = bv_raw.iloc[0] if len(bv_raw) else np.nan
            v_disp     = _apply(transform, raw_v)
            base_disp  = _apply(transform, base_raw)
            out[disp] = v_disp
            if pd.notna(v_disp) and pd.notna(base_disp) and base_disp != 0:
                out[f"{disp} | VsRef"] = (v_disp / base_disp) * 100.0
            else:
                out[f"{disp} | VsRef"] = np.nan
        rows.append(out)
    df = pd.DataFrame(rows)
    direction_lookup = {disp: d for disp, _, d, _ in resolved}
    return df, direction_lookup


def _write_color_coded_sheet(writer, sheet_name: str, df: pd.DataFrame,
                             baseline_id: str, group_col: str = 'Electrolyte',
                             direction_lookup=None):
    if direction_lookup is None:
        direction_lookup = {}
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    wb = writer.book
    ws = writer.sheets[sheet_name]

    bold = wb.add_format({'bold': True, 'border': 1, 'align': 'center',
                          'bg_color': '#404040', 'font_color': 'white'})
    for col_idx, col_name in enumerate(df.columns):
        ws.write(0, col_idx, str(col_name), bold)
    ws.set_column(0, 0, 14)
    if 'n' in df.columns:
        ws.set_column(df.columns.get_loc('n'), df.columns.get_loc('n'), 6)
    for col_idx, c in enumerate(df.columns):
        if c not in (group_col, 'n'):
            ws.set_column(col_idx, col_idx, 13)

    fmt        = wb.add_format({'num_format': '0.0',  'border': 1})
    fmt_int    = wb.add_format({'num_format': '0',    'border': 1, 'align': 'center'})
    fmt_dim    = wb.add_format({'num_format': '0.0',  'border': 1, 'font_color': '#999999'})
    n_rows = len(df)
    n_col_idx = df.columns.get_loc('n') if 'n' in df.columns else None
    for r in range(n_rows):
        n_cells = int(df.iloc[r, n_col_idx]) if n_col_idx is not None else 99
        dim = (n_cells < 2)
        for col_idx, col_name in enumerate(df.columns):
            if col_idx == 0:
                continue
            v = df.iloc[r, col_idx]
            if col_name == 'n':
                ws.write_number(r+1, col_idx, int(v) if pd.notna(v) else 0, fmt_int)
            elif pd.isna(v):
                ws.write(r+1, col_idx, '', fmt)
            else:
                ws.write_number(r+1, col_idx, float(v), fmt_dim if dim else fmt)

    for col_idx, col_name in enumerate(df.columns):
        if 'VsRef' not in str(col_name):
            continue
        disp = str(col_name).replace(' | VsRef', '').strip()
        direction = direction_lookup.get(disp, +1)
        if direction == +1:
            min_color, mid_color, max_color = '#F4A8A8', '#FFFFFF', '#9EE39E'
        else:
            min_color, mid_color, max_color = '#9EE39E', '#FFFFFF', '#F4A8A8'
        first = f'{_xl_col(col_idx)}2'
        last  = f'{_xl_col(col_idx)}{n_rows+1}'
        ws.conditional_format(f'{first}:{last}', {
            'type':      '3_color_scale',
            'min_type':  'num', 'min_value':  85, 'min_color':  min_color,
            'mid_type':  'num', 'mid_value': 100, 'mid_color':  mid_color,
            'max_type':  'num', 'max_value': 115, 'max_color':  max_color,
        })

    ws.freeze_panes(1, 1)


def _xl_col(idx: int) -> str:
    s = ''
    n = idx
    while True:
        s = chr(ord('A') + (n % 26)) + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


# ---------------------------------------------------------------------------
# Direction inference
# ---------------------------------------------------------------------------

_LOWER_IS_BETTER_PATTERNS = [
    'dcir',          'fade',           'fading',         'cap_loss',
    'thickness',     'acr',            'kinetic_fading', 'kinetic fading',
    'v_slippage',    'v_slip',         'v_polariz',      'cum_ce_loss',
    'trapped_li',    'abruptness',     'voltage_drop',   'side_react',
    'growth',        'resistance',     'polarization',   'mohm',
]
_HIGHER_IS_BETTER_PATTERNS = [
    'cycle_life',   'retention',     'discharge_capacity',  'charge_capacity',
    '_capacity',    'cap_full',      'rce',                 'coulombic',
    'ce_ex1',       'ce_avg',        'ce_pct',              'first_ce',
    'efficiency',   'rate_cap_ratio','energy_efficiency',
]


def infer_feature_direction(name: str) -> int:
    s = str(name).lower()
    for p in _LOWER_IS_BETTER_PATTERNS:
        if p in s:
            return -1
    for p in _HIGHER_IS_BETTER_PATTERNS:
        if p in s:
            return +1
    return +1


def _write_combined_pct_delta_sheet(writer, sheet_name: str,
                                    pct_df: pd.DataFrame,
                                    role_map: dict,
                                    group_col: str = 'Electrolyte'):
    pct_df.to_excel(writer, sheet_name=sheet_name, index=False)
    wb = writer.book
    ws = writer.sheets[sheet_name]

    hdr_in  = wb.add_format({'bold': True, 'border': 1, 'align': 'center',
                             'bg_color': '#2E5C8A', 'font_color': 'white'})
    hdr_out = wb.add_format({'bold': True, 'border': 1, 'align': 'center',
                             'bg_color': '#8A2E5C', 'font_color': 'white'})
    hdr_meta = wb.add_format({'bold': True, 'border': 1, 'align': 'center',
                              'bg_color': '#404040', 'font_color': 'white'})
    fmt_num = wb.add_format({'num_format': '0.0',     'border': 1})
    fmt_sci = wb.add_format({'num_format': '0.00E+00','border': 1})
    for col_idx, col_name in enumerate(pct_df.columns):
        if col_idx == 0 or col_name in ('n_cells_used', '_is_baseline'):
            continue
        col_vals = pd.to_numeric(pct_df.iloc[:, col_idx], errors='coerce').dropna()
        col_max = col_vals.abs().max() if len(col_vals) else 0
        chosen = fmt_sci if (col_max > 0 and col_max < 0.1) else fmt_num
        for r in range(len(pct_df)):
            v = pct_df.iloc[r, col_idx]
            if pd.isna(v):
                continue
            ws.write_number(r+1, col_idx, float(v), chosen)

    n_rows = len(pct_df)
    for col_idx, col_name in enumerate(pct_df.columns):
        role = role_map.get(col_name, None)
        if col_idx == 0 or col_name in ('n_cells_used', '_is_baseline'):
            ws.write(0, col_idx, str(col_name), hdr_meta)
        elif role == 'INPUT':
            ws.write(0, col_idx, str(col_name) + '  [IN]', hdr_in)
        elif role == 'OUTPUT':
            ws.write(0, col_idx, str(col_name) + '  [OUT]', hdr_out)
        else:
            ws.write(0, col_idx, str(col_name), hdr_meta)
        if col_idx == 0:
            ws.set_column(0, 0, 14)
            continue
        if col_name in ('n_cells_used', '_is_baseline'):
            ws.set_column(col_idx, col_idx, 11)
            continue
        ws.set_column(col_idx, col_idx, 13)
        direction = infer_feature_direction(col_name)
        if direction == +1:
            min_color, mid_color, max_color = '#F4A8A8', '#FFFFFF', '#9EE39E'
            min_v, max_v = -15, 15
        else:
            min_color, mid_color, max_color = '#9EE39E', '#FFFFFF', '#F4A8A8'
            min_v, max_v = -15, 15
        first = f'{_xl_col(col_idx)}2'
        last  = f'{_xl_col(col_idx)}{n_rows+1}'
        ws.conditional_format(f'{first}:{last}', {
            'type':      '3_color_scale',
            'min_type':  'num', 'min_value':  min_v, 'min_color': min_color,
            'mid_type':  'num', 'mid_value':  0,     'mid_color': mid_color,
            'max_type':  'num', 'max_value':  max_v, 'max_color': max_color,
        })

    ws.freeze_panes(1, 1)


# ---------------------------------------------------------------------------
# Variability sheet
# ---------------------------------------------------------------------------

def build_variability_sheet(lot_df: pd.DataFrame,
                             std_df: pd.DataFrame,
                             group_col: str = 'Electrolyte') -> pd.DataFrame:
    feat_cols = [c for c in lot_df.columns if c not in (group_col, 'n_cells_used')
                 and pd.api.types.is_numeric_dtype(lot_df[c])]
    out = lot_df[[group_col]].copy()
    if 'n_cells_used' in lot_df.columns:
        out['n_cells_used'] = lot_df['n_cells_used'].values
    for c in feat_cols:
        m = lot_df[c].astype(float).abs()
        s = std_df[c].astype(float) if c in std_df.columns else np.nan
        out[c] = np.where(m > 0, (s / m) * 100.0, np.nan)
    return out


# ---------------------------------------------------------------------------
# Sigma-delta sheet
# ---------------------------------------------------------------------------

def _write_sigma_delta_coloured(writer, sheet_name: str, sigma_df: pd.DataFrame,
                                  group_col: str = 'Electrolyte'):
    sigma_df.to_excel(writer, sheet_name=sheet_name, index=False)
    wb = writer.book; ws = writer.sheets[sheet_name]
    n_rows = len(sigma_df)
    feat_cols = [c for c in sigma_df.columns
                 if c not in (group_col, 'n_cells_used', '_is_baseline')
                 and pd.api.types.is_numeric_dtype(sigma_df[c])]
    for c in feat_cols:
        col_idx = list(sigma_df.columns).index(c)
        first = f'{_xl_col(col_idx)}2'
        last  = f'{_xl_col(col_idx)}{n_rows+1}'
        rng = f'{first}:{last}'
        ws.conditional_format(rng, {
            'type':      '3_color_scale',
            'min_type':  'num', 'min_value': -3, 'min_color': '#9EE39E',
            'mid_type':  'num', 'mid_value':  0, 'mid_color': '#FFFFFF',
            'max_type':  'num', 'max_value': +3, 'max_color': '#F4A8A8',
        })
    fmt = wb.add_format({'num_format': '0.0'})
    for ci, c in enumerate(sigma_df.columns):
        if c in feat_cols:
            ws.set_column(ci, ci, 10, fmt)
    ws.freeze_panes(1, 1)


# ---------------------------------------------------------------------------
# Bad-cell explanation
# ---------------------------------------------------------------------------

def explain_bad_cells(df_z_cells: pd.DataFrame,
                       flag_df: pd.DataFrame,
                       z_threshold: float = 5.0,
                       max_features_per_cell: int = 10,
                       id_col: str = 'Barcode') -> pd.DataFrame:
    if 'is_bad_cell' not in flag_df.columns:
        return flag_df
    bad = flag_df[flag_df['is_bad_cell']].copy() if flag_df['is_bad_cell'].any() else flag_df.head(0).copy()
    if bad.empty:
        bad['offending_features'] = ''
        return bad
    z_indexed = df_z_cells.set_index(id_col) if id_col in df_z_cells.columns else df_z_cells
    explanations = []
    for bc in bad[id_col].astype(str):
        if bc not in z_indexed.index.astype(str).tolist():
            explanations.append(''); continue
        row = z_indexed.loc[bc]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        offenders = []
        for f, z in row.items():
            try:
                z_val = float(z)
            except Exception:
                continue
            if abs(z_val) >= z_threshold:
                offenders.append((f, z_val))
        offenders.sort(key=lambda kv: -abs(kv[1]))
        offenders = offenders[:max_features_per_cell]
        explanations.append('; '.join([f'{f}={z:+.1f}sigma' for f, z in offenders]))
    bad['offending_features'] = explanations
    return bad


# ---------------------------------------------------------------------------
# Per-test chartsheets
# ---------------------------------------------------------------------------

def _write_per_test_charts(writer, color_summary: pd.DataFrame,
                             direction_lookup: dict,
                             group_col: str = 'Electrolyte'):
    wb = writer.book
    if color_summary.empty: return

    feat_cols = [c for c in color_summary.columns
                 if c not in (group_col, 'n') and 'VsRef' not in str(c)]
    chartsheets = {}
    for c in feat_cols:
        safe = re.sub(r'[^A-Za-z0-9_-]+', '_', str(c))[:24]
        chartsheets[c] = wb.add_chartsheet(f'Chart_{safe}'[:31])

    ws_name = 'color_coded_summary'
    n_rows = len(color_summary)
    cols = list(color_summary.columns)
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2",
               "#7f7f7f","#bcbd22","#17becf","#aec7e8","#ffbb78","#98df8a","#ff9896",
               "#c5b0d5","#c49c94","#f7b6d2","#dbdb8d","#9edae5","#393b79"]
    el_color = {str(e): palette[i % len(palette)]
                for i, e in enumerate(color_summary[group_col].astype(str))}
    el_idx = cols.index(group_col)
    for c in feat_cols:
        ci = cols.index(c)
        ch = wb.add_chart({'type': 'column'})
        pts = [{'fill': {'color': el_color.get(str(e), '#1f77b4')},
                'border':{'color': el_color.get(str(e), '#1f77b4')}}
               for e in color_summary[group_col]]
        ch.add_series({
            'categories': [ws_name, 1, el_idx, n_rows, el_idx],
            'values':     [ws_name, 1, ci,     n_rows, ci],
            'name':       c,
            'points':     pts,
            'data_labels':{'value': True, 'num_format': '0.0'},
        })
        ch.set_title({'name': c, 'name_font': {'name':'Helvetica Neue','size':14,'bold':True}})
        ch.set_x_axis({'num_font': {'name':'Helvetica Neue','size':10,'rotation':-45}})
        ch.set_y_axis({'name': c, 'name_font': {'name':'Helvetica Neue','size':12,'bold':True}})
        ch.set_legend({'none': True})
        chartsheets[c].set_chart(ch)


# ---------------------------------------------------------------------------
# Top-level Stage 2 driver
# ---------------------------------------------------------------------------

_NAME_SUFFIX_REGEX = re.compile(r'[-_](A|B|C|v\d|V\d|new|old)$', re.IGNORECASE)


def normalize_electrolyte_name(name) -> str:
    s = str(name).strip()
    s = _NAME_SUFFIX_REGEX.sub('', s)
    return s


def run_stage2(
    doe_root: str,
    baseline_electrolyte: str,
    output_dir=None,
    doe_name=None,
    scaler_path=None,
    refit_scaler_if_missing: bool = False,
    z_threshold: float = 5.0,
    extreme_count_threshold: int = 10,
    normalize_electrolyte_names: bool = True,
    name_overrides=None,
    verbose: bool = True,
) -> dict:
    doe_root = Path(doe_root)
    if output_dir is None:
        output_dir = doe_root
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    if doe_name is None:
        doe_name = doe_root.name

    stage1_map = discover_stage1_folders(str(doe_root))
    if not stage1_map:
        raise FileNotFoundError(f"No stage1_* folders in {doe_root}")
    if verbose:
        print(f"[stage2] discovered {len(stage1_map)} stage1 folders: {list(stage1_map.keys())}")

    _names = list(stage1_map.keys())
    for i, a in enumerate(_names):
        for b in _names[i+1:]:
            if (a.replace('_','') == b.replace('_','') or a.lower() == b.lower()):
                print(f"[stage2] WARNING near-duplicate stage1 folder names: '{a}' and '{b}'.")

    name_overrides = name_overrides or {}

    per_test_dfs = {}
    role_map_all = {}
    test_meta = {}
    cycling_barcodes = set()
    raw_name_set: dict = {}
    for test_item, path in stage1_map.items():
        try:
            df, meta = _load_stage1(path)
        except Exception as e:
            if verbose: print(f"[stage2]   {test_item}: load failed ({e}), skip")
            continue
        if df is None or len(df) == 0:
            continue
        test_type = meta.get('test_type', 'cycling')

        df = df.copy()
        df['Electrolyte_raw'] = df['Electrolyte'].astype(str)
        if normalize_electrolyte_names:
            df['Electrolyte'] = df['Electrolyte_raw'].map(normalize_electrolyte_name)
        if name_overrides:
            df['Electrolyte'] = df['Electrolyte'].replace(name_overrides)

        raw_name_set[test_item] = sorted(df['Electrolyte_raw'].unique().tolist())
        df = df.drop(columns=['Electrolyte_raw'])

        df_tagged, rmap = _tag_columns(df, test_item, test_type)
        per_test_dfs[test_item] = df_tagged
        role_map_all.update(rmap)
        test_meta[test_item] = test_type
        if verbose:
            print(f"[stage2]   {test_item}  type={test_type}  {df.shape[0]} cells x {df.shape[1]-2} feats")
        if test_type in ('cycling', 'cycle'):
            cycling_barcodes.update(df['Barcode'].astype(str).tolist())

    if cycling_barcodes:
        for ti, df in list(per_test_dfs.items()):
            if test_meta[ti] == 'formation' and len(df) > len(cycling_barcodes) * 1.2:
                before = len(df)
                df = df[df['Barcode'].astype(str).isin(cycling_barcodes)].reset_index(drop=True)
                per_test_dfs[ti] = df
                if verbose:
                    print(f"[stage2]   filtered {ti}: {before} -> {len(df)} cells (DOE-specific)")

    keys = ['Barcode', 'Electrolyte']
    df_all = None
    for ti, df in per_test_dfs.items():
        df_all = df if df_all is None else df_all.merge(df, on=keys, how='outer')

    df_all = df_all[df_all['Barcode'].astype(str).str.match(r'^B\d+', na=False)]
    df_all = df_all[~df_all['Electrolyte'].astype(str).str.lower().isin(('nan', 'none', ''))]
    df_all = df_all.reset_index(drop=True)
    if verbose:
        print(f"[stage2] merged: {df_all.shape[0]} cells x {df_all.shape[1]-2} features")
        if normalize_electrolyte_names:
            unique_after = sorted(df_all['Electrolyte'].dropna().unique().tolist())
            print(f"[stage2] electrolytes after normalization ({len(unique_after)}): {unique_after}")

    if scaler_path is None:
        scaler_path = str(output_dir / 'feature_scales.json')
    if os.path.exists(scaler_path):
        scaler = FeatureScaler.load(scaler_path)
        if verbose: print(f"[stage2] LOADED frozen scaler -> {scaler_path}  ({len(scaler.params)} features)")
    elif refit_scaler_if_missing:
        scaler = FeatureScaler(center='global_median').fit(df_all)
        scaler.save(scaler_path)
        if verbose:
            print(f"[stage2] FIT new scaler (first DOE for this chemistry/format)")
            print(f"           -> saved to {scaler_path}")
    else:
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path} and refit_scaler_if_missing=False.\n"
            f"For the FIRST DOE of a new chemistry/format, set refit_scaler_if_missing=True."
        )

    df_z_cells = scaler.transform(df_all)
    flag_df = flag_bad_cells(df_z_cells, id_col='Barcode',
                              z_threshold=z_threshold,
                              extreme_count_threshold=extreme_count_threshold)
    bad_set = set(flag_df.loc[flag_df['is_bad_cell'], 'Barcode'].astype(str).tolist())
    if bad_set:
        if verbose:
            print(f"[stage2] bad cells: {sorted(bad_set)}")
        clean = df_all[~df_all['Barcode'].astype(str).isin(bad_set)].copy()
    else:
        clean = df_all.copy()

    lot_df = aggregate_cells_to_lots(clean)
    lot_std_df = aggregate_cells_to_lots_std(clean)

    pct_delta = compute_delta_vs_baseline(lot_df, baseline_electrolyte, mode='pct')

    lot_df_z = scaler.transform(lot_df)
    sigma_delta = compute_delta_vs_baseline(lot_df_z, baseline_electrolyte, mode='subtract')

    out_xlsx = output_dir / f"stage2_{doe_name}.xlsx"
    out_pq   = output_dir / f"stage2_{doe_name}.parquet"

    # Excel-lock fallback: if user has stage2_<DOE>.xlsx open in Excel, the
    # write will fail with PermissionError.  Detect and write to a _auto suffix.
    def _is_locked(p):
        if not Path(p).exists(): return False
        try:
            with open(p, 'a'):
                return False
        except Exception:
            return True
    if _is_locked(out_xlsx):
        out_xlsx = output_dir / f"stage2_{doe_name}_auto.xlsx"
        if verbose:
            print(f"[stage2] {doe_name}.xlsx is locked (open in Excel?). "
                  f"Writing to {out_xlsx.name} instead.")

    color_summary, dir_lookup = build_color_coded_summary(
        lot_df, pct_delta, baseline_id=baseline_electrolyte
    )

    variability_df = build_variability_sheet(lot_df, lot_std_df)

    bad_cell_explained = explain_bad_cells(df_z_cells, flag_df,
                                            z_threshold=z_threshold)

    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as w:
        if not color_summary.empty:
            _write_color_coded_sheet(w, 'color_coded_summary',
                                     color_summary,
                                     baseline_id=baseline_electrolyte,
                                     direction_lookup=dir_lookup)
        if not color_summary.empty:
            _write_per_test_charts(w, color_summary, dir_lookup)
        _write_combined_pct_delta_sheet(w, 'pct_delta',  pct_delta,  role_map_all)
        _write_sigma_delta_coloured(w, 'sigma_delta', sigma_delta)
        lot_df.to_excel(w,        sheet_name='lot_avg',        index=False)
        variability_df.to_excel(w, sheet_name='lot_variability_CV', index=False)
        lot_std_df.to_excel(w,    sheet_name='lot_std',        index=False)
        bad_cell_explained.to_excel(w, sheet_name='bad_cell_flags', index=False)
        pd.DataFrame([
            {'test_item': k, 'test_type': v} for k, v in test_meta.items()
        ]).to_excel(w, sheet_name='test_items', index=False)
        pd.DataFrame([
            {'feature': k, 'role': v} for k, v in role_map_all.items()
        ]).to_excel(w, sheet_name='role_map', index=False)

    parquet_df = sigma_delta.copy()
    parquet_df.attrs = {}
    try:
        parquet_df.to_parquet(out_pq, index=False)
    except Exception:
        parquet_df.to_pickle(str(out_pq).replace('.parquet', '.pkl'))

    if verbose:
        print(f"[stage2] wrote {out_xlsx}")
        print(f"[stage2] wrote {out_pq}")

    return {
        'doe_name':         doe_name,
        'test_meta':        test_meta,
        'role_map':         role_map_all,
        'merged_cells_df':  df_all,
        'clean_df':         clean,
        'flag_df':          flag_df,
        'lot_df':           lot_df,
        'pct_delta':        pct_delta,
        'sigma_delta':      sigma_delta,
        'color_summary':    color_summary,
        'output_xlsx':      str(out_xlsx),
        'output_parquet':   str(out_pq),
        'scaler_path':      scaler_path,
    }

