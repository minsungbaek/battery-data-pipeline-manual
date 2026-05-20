"""Utility helpers for Cycling_Processor notebooks.

The notebook keeps the interactive workflow and Excel-writing order, while this
module holds reusable parsing, feature extraction, chart, and table helpers.
"""

from __future__ import annotations

import re
from tkinter import Tk, filedialog

import numpy as np
import pandas as pd

try:
    import traitlets
    from ipywidgets import widgets
except ModuleNotFoundError:
    traitlets = None
    widgets = None


__all__ = [
    "get_statis",
    "SelectFilesButton",
    "SelectDirectoryButton",
    "prune_legend",
    "make_safe_sheet_name",
    "make_safe_chart_sheet_name",
    "adjust_time",
    "as_float64_series",
    "safe_divide",
    "add_1c_cycle_excl_first",
    "load_cycle_map",
    "attach_cycle_map_info",
    "col_idx",
    "add_series_by_col",
    "clean_series",
    "update_range",
    "update_max",
    "add_df_series",
    "set_chart_axis",
    "build_dcir_subset",
    "get_exact_cycle_value",
    "get_end_minus_one_value",
    "calc_slope_between_points",
    "calc_slope_1_to_endm1",
    "calc_abruptness",
    "get_final_1c_cycle_number",
    "get_end_minus_n_value",
    "calc_slope_if_valid",
    "extract_1c_features_extended_from_df",
    "average_between_cycles",
    "extract_ce_like_features",
    "safe_pct_delta",
    "safe_rate_per_cycle",
    "build_c10_summary_row",
    "build_full_cell_table",
    "build_1c_ex1_table",
    "build_c10_combined_table",
]


def get_statis(df_detail: pd.DataFrame) -> pd.DataFrame:
    """Build the Neware step-level statistics table from detail data."""
    df_detail_group = df_detail.groupby("Step")
    df_statis = pd.concat([
        df_detail_group["Cycle"].last().to_frame(),
        df_detail_group["Current(mA)"].first().to_frame(),
        df_detail_group["Current(mA)"].last().to_frame(),
        df_detail_group["Voltage"].first().to_frame(),
        df_detail_group["Voltage"].last().to_frame(),
        df_detail_group["Status"].first().to_frame(),
        df_detail_group["Time"].last().to_frame(),
        df_detail_group["Charge_Capacity(mAh)"].last().to_frame(),
        df_detail_group["Discharge_Capacity(mAh)"].last().to_frame(),
        df_detail_group["Charge_Energy(mWh)"].last().to_frame(),
        df_detail_group["Discharge_Energy(mWh)"].last().to_frame(),
        df_detail_group["Timestamp"].first().to_frame(),
        df_detail_group["Timestamp"].last().to_frame(),
    ], axis=1)
    df_statis.columns = [
        "Cycle",
        "Start Current(mA)", "End Current(mA)",
        "Start Voltage(V)", "End Voltage(V)",
        "Status", "Step time",
        "Charge Capacity(mAh)", "Discharge Capacity(mAh)",
        "Charge Energy(mWh)", "Discharge Energy(mWh)",
        "Start time", "End time",
    ]
    df_statis.reset_index(inplace=True)
    df_statis["Last End Voltage(V)"] = df_statis["End Voltage(V)"].shift()
    df_statis["DCIR(Ohm)"] = 1000 * (
        (df_statis["Start Voltage(V)"] - df_statis["Last End Voltage(V)"])
        / df_statis["Start Current(mA)"]
    ).abs()
    return df_statis


if widgets is not None and traitlets is not None:

    class SelectFilesButton(widgets.Button):
        """ipywidgets button backed by tkinter's multi-file picker."""

        def __init__(self, output=None):
            super().__init__(description="Select Files", icon="square-o")
            self.add_traits(files=traitlets.traitlets.List())
            self.output_area = output
            self.style.button_color = "orange"
            self.on_click(self.select_files)

        @staticmethod
        def select_files(button):
            def choose_files():
                root = Tk()
                root.withdraw()
                root.call("wm", "attributes", ".", "-topmost", True)
                button.files = filedialog.askopenfilename(multiple=True)
                button.description = "Files Selected"
                button.icon = "check-square-o"
                button.style.button_color = "lightgreen"

            try:
                if getattr(button, "output_area", None) is not None:
                    with button.output_area:
                        choose_files()
                else:
                    choose_files()
            except Exception:
                pass


    class SelectDirectoryButton(widgets.Button):
        """ipywidgets button backed by tkinter's directory picker."""

        def __init__(self, output=None):
            super().__init__(description="Select Directory", icon="square-o")
            self.add_traits(current_directory=traitlets.traitlets.Unicode())
            self.output_area = output
            self.style.button_color = "orange"
            self.on_click(self.select_directory)

        @staticmethod
        def select_directory(button):
            def choose_directory():
                root = Tk()
                root.withdraw()
                root.call("wm", "attributes", ".", "-topmost", True)
                button.current_directory = filedialog.askdirectory()
                button.description = "Directory Selected"
                button.icon = "check-square-o"
                button.style.button_color = "lightgreen"

            try:
                if getattr(button, "output_area", None) is not None:
                    with button.output_area:
                        choose_directory()
                else:
                    choose_directory()
            except Exception:
                pass

else:

    class SelectFilesButton:
        """Placeholder used when notebook widget dependencies are unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("Install ipywidgets and traitlets to use file picker widgets.")


    class SelectDirectoryButton:
        """Placeholder used when notebook widget dependencies are unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError("Install ipywidgets and traitlets to use directory picker widgets.")



# function to prune the series so the legend has only 1 bar/group and not 1 line/cell
def prune_legend(metadata):
    positions = []
    index = pd.Index(metadata.Group)
    groups = metadata.Group.unique()
    for group in groups:
        position = index.get_loc(group)
        if isinstance(position, slice):
            positions.append(position.start)
        elif isinstance(position, np.ndarray):
            positions.append(np.where(position)[0].tolist()[0])
        else:
            positions.append(position)
    return list(set(range(len(index))) - set(positions))


def make_safe_sheet_name(name: str, max_length: int = 31) -> str:
    invalid_chars = r'[\\/*?:[\]]'
    safe_name = re.sub(invalid_chars, "_", name)
    return safe_name[:max_length] if len(safe_name) > max_length else safe_name


def make_safe_chart_sheet_name(name: str, max_length: int = 31) -> str:
    return make_safe_sheet_name(name, max_length)


def adjust_time(df, time_col, new_time_col):
    time_diff = np.diff(df[time_col], prepend=0)
    reset_idx = time_diff < 0
    if True in reset_idx:
        increments = df[time_col][reset_idx] - time_diff[reset_idx]
        true_indices = np.where(reset_idx)[0]
        new_array = np.zeros(reset_idx.shape, dtype=float)
        new_array[true_indices] = increments
        adjustment = np.cumsum(new_array)
        df[new_time_col] = df[time_col] + adjustment
    else:
        df[new_time_col] = df[time_col]
    return df


def as_float64_series(s):
    return pd.to_numeric(s, errors='coerce').astype('float64')


def safe_divide(numer, denom, fill_value=np.nan):
    """
    Return float64 numpy array from numer/denom with safe handling.
    """
    numer_arr = pd.to_numeric(numer, errors='coerce').to_numpy(dtype=np.float64)
    denom_arr = pd.to_numeric(denom, errors='coerce').to_numpy(dtype=np.float64)

    out = np.full(len(numer_arr), fill_value, dtype=np.float64)
    valid = np.isfinite(denom_arr) & (denom_arr != 0)
    np.divide(numer_arr, denom_arr, out=out, where=valid)
    return out


def add_1c_cycle_excl_first(cycle_map_df):
    """
    Create a new cycle column that excludes the first 1C point
    of each contiguous 1C block, while keeping a GLOBAL continuous numbering.
    """
    cm = cycle_map_df.copy()

    cm['Corrected Cycle'] = pd.to_numeric(cm['Corrected Cycle'], errors='coerce')
    cm['1C Cycle'] = pd.to_numeric(cm['1C Cycle'], errors='coerce')
    cm['NW Cycle'] = pd.to_numeric(cm['NW Cycle'], errors='coerce')

    cm = cm.sort_values(['Corrected Cycle', 'NW Cycle']).reset_index(drop=True)

    new_vals = [np.nan] * len(cm)
    valid_idx = cm.index[cm['1C Cycle'].notna()].tolist()

    if len(valid_idx) == 0:
        cm['1C Cycle Excl First'] = np.nan
        return cm

    global_counter = 0
    prev_corr = None

    for idx in valid_idx:
        curr_corr = cm.loc[idx, 'Corrected Cycle']
        is_new_block = (prev_corr is None) or (curr_corr != prev_corr + 1)

        if is_new_block:
            new_vals[idx] = np.nan
        else:
            global_counter += 1
            new_vals[idx] = global_counter

        prev_corr = curr_corr

    cm['1C Cycle Excl First'] = new_vals
    return cm


def load_cycle_map(cycle_map_path):
    cm = pd.read_excel(cycle_map_path).rename(columns={
        'NW cycles': 'NW Cycle',
        'Correct cycles': 'Corrected Cycle',
        '1C_cycles': '1C Cycle',
        'note': 'Note',
        'State': 'State'
    })
    cm = cm[['NW Cycle', 'Corrected Cycle', '1C Cycle', 'Note', 'State']].copy()

    for c in ['NW Cycle', 'Corrected Cycle', '1C Cycle']:
        cm[c] = pd.to_numeric(cm[c], errors='coerce')

    for c in ['Note', 'State']:
        cm[c] = cm[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    cm = cm.dropna(subset=['NW Cycle']).copy()
    cm = add_1c_cycle_excl_first(cm)
    return cm


def attach_cycle_map_info(df, cycle_map_df):
    df = df.copy()
    df['Cycle'] = pd.to_numeric(df['Cycle'], errors='coerce')
    cm = cycle_map_df.copy()
    cm['NW Cycle'] = pd.to_numeric(cm['NW Cycle'], errors='coerce')
    cm_map = cm.drop_duplicates(subset=['NW Cycle']).copy()

    df['1C Cycle'] = df['Cycle'].map(dict(zip(cm_map['NW Cycle'], cm_map['1C Cycle'])))
    df['1C Cycle Excl First'] = df['Cycle'].map(dict(zip(cm_map['NW Cycle'], cm_map['1C Cycle Excl First'])))
    df['Cycle Note'] = df['Cycle'].map(dict(zip(cm_map['NW Cycle'], cm_map['Note'])))
    df['Cycle State'] = df['Cycle'].map(dict(zip(cm_map['NW Cycle'], cm_map['State'])))
    df['Map Corrected Cycle'] = df['Cycle'].map(dict(zip(cm_map['NW Cycle'], cm_map['Corrected Cycle'])))
    return df


def col_idx(df, col_name):
    return df.columns.get_loc(col_name)


def add_series_by_col(chart, sheet_name, nrows, x_col, y_col, color, name):
    if nrows <= 0:
        return
    chart.add_series({
        'categories': [sheet_name, 1, x_col, nrows, x_col],
        'values': [sheet_name, 1, y_col, nrows, y_col],
        'line': {'color': color},
        'name': name,
    })


def clean_series(s):
    return pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()


def update_range(ranges, key, s):
    s = clean_series(s)
    if len(s) == 0:
        return
    ranges[key]['min'] = min(ranges[key]['min'], s.min())
    ranges[key]['max'] = max(ranges[key]['max'], s.max())


def update_max(max_tracker, key, value):
    if pd.notna(value):
        max_tracker[key] = float(max(max_tracker.get(key, 0), value))

def add_df_series(chart, df, sheet_name, x_col_name, y_col_name, color, name):
    if len(df) == 0 or x_col_name not in df.columns or y_col_name not in df.columns:
        return
    add_series_by_col(
        chart, sheet_name, len(df),
        col_idx(df, x_col_name),
        col_idx(df, y_col_name),
        color, name
    )


def set_chart_axis(chart, x_name, y_name, x_min, x_max, y_min, y_max, remove_legends):
    chart.set_x_axis({
        'name': x_name,
        'name_font': {'name': 'Helvetica Neue', 'size': 14, 'bold': True},
        'min': x_min,
        'max': x_max,
    })
    chart.set_y_axis({
        'name': y_name,
        'name_font': {'name': 'Helvetica Neue', 'size': 14, 'bold': True},
        'min': y_min,
        'max': y_max,
    })
    chart.set_legend({'delete_series': remove_legends})


def build_dcir_subset(df_dcir, include_dcir0=False):
    if len(df_dcir) == 0:
        return None
    out = df_dcir.copy()
    out['nDCIR'] = out['DCIR'] / out['DCIR'].iloc[0]
    if include_dcir0 and 'DCIR0' in out.columns and 'DCIRt' in out.columns:
        out['nDCIR0'] = out['DCIR0'] / out['DCIR0'].iloc[0]
        out['nDCIRt'] = out['DCIRt'] / out['DCIRt'].iloc[0]
    return out


# =========================
# 1C feature extraction helpers
# =========================
def get_exact_cycle_value(s, cycle_target, cycle_col='1C Cycle Excl First'):
    temp = s.loc[s[cycle_col] == cycle_target, 'y']
    if len(temp) == 0:
        return np.nan
    val = pd.to_numeric(temp, errors='coerce').iloc[0]
    return val if pd.notna(val) else np.nan


def get_end_minus_one_value(s, cycle_col='1C Cycle Excl First'):
    if len(s) == 0:
        return np.nan
    max_cycle = s[cycle_col].max()
    if pd.isna(max_cycle):
        return np.nan
    target = max_cycle - 1
    if target < 1:
        return np.nan
    val = get_exact_cycle_value(s, target, cycle_col=cycle_col)
    if pd.isna(val):
        return get_exact_cycle_value(s, max_cycle, cycle_col=cycle_col)
    return val


def calc_slope_between_points(s, cycle_start, cycle_end, cycle_col='1C Cycle Excl First'):
    if len(s) == 0:
        return np.nan

    max_cycle = s[cycle_col].max()
    if pd.isna(max_cycle):
        return np.nan

    cycle_end_safe = min(cycle_end, max_cycle)
    if cycle_end_safe <= cycle_start:
        return np.nan

    y1 = get_exact_cycle_value(s, cycle_start, cycle_col=cycle_col)
    y2 = get_exact_cycle_value(s, cycle_end_safe, cycle_col=cycle_col)

    if pd.isna(y1) or pd.isna(y2):
        return np.nan

    return (y2 - y1) / (cycle_end_safe - cycle_start)


def calc_slope_1_to_endm1(s, cycle_col='1C Cycle Excl First'):
    if len(s) == 0:
        return np.nan

    max_cycle = s[cycle_col].max()
    if pd.isna(max_cycle):
        return np.nan

    endm1 = max_cycle - 1
    if endm1 <= 1:
        endm1 = max_cycle

    if endm1 <= 1:
        return np.nan

    return calc_slope_between_points(s, 1, endm1, cycle_col=cycle_col)


def calc_abruptness(s, cycle_col='1C Cycle Excl First'):
    if len(s) == 0:
        return np.nan

    s2 = s.copy()
    s2 = s2[(s2[cycle_col] >= 1)].copy()
    if len(s2) < 2:
        return np.nan

    max_cycle = s2[cycle_col].max()
    if pd.isna(max_cycle):
        return np.nan

    endm1 = max_cycle - 1
    if endm1 < 1:
        endm1 = max_cycle

    s2 = s2[s2[cycle_col] <= endm1].copy()
    if len(s2) == 0:
        return np.nan

    f1 = get_exact_cycle_value(s2, 1, cycle_col=cycle_col)
    fend = get_exact_cycle_value(s2, endm1, cycle_col=cycle_col)

    if pd.isna(f1) or pd.isna(fend):
        return np.nan

    denom = f1 - fend
    if pd.isna(denom) or denom == 0:
        return np.nan

    z = (pd.to_numeric(s2['y'], errors='coerce') - fend) / denom
    z = z.replace([np.inf, -np.inf], np.nan).dropna()

    if len(z) == 0:
        return np.nan

    return 1 - 2 * z.mean()


def get_final_1c_cycle_number(df_in, cycle_col='1C Cycle'):
    if cycle_col not in df_in.columns or len(df_in) == 0:
        return np.nan
    s = pd.to_numeric(df_in[cycle_col], errors='coerce').dropna()
    if len(s) == 0:
        return np.nan
    return s.max()


def get_end_minus_n_value(s, n, cycle_col='1C Cycle Excl First'):
    if len(s) == 0:
        return np.nan

    max_cycle = s[cycle_col].max()
    if pd.isna(max_cycle):
        return np.nan

    target = max_cycle - n
    if target < 1:
        return np.nan

    return get_exact_cycle_value(s, target, cycle_col=cycle_col)


def calc_slope_if_valid(s, cycle_start, cycle_end, cycle_col='1C Cycle Excl First', require_max_cycle_at_least=None):
    if len(s) == 0:
        return np.nan

    max_cycle = s[cycle_col].max()
    if pd.isna(max_cycle):
        return np.nan

    if require_max_cycle_at_least is not None and max_cycle < require_max_cycle_at_least:
        return np.nan

    if cycle_start < 1 or cycle_end < 1:
        return np.nan

    return calc_slope_between_points(s, cycle_start, cycle_end, cycle_col=cycle_col)


def extract_1c_features_extended_from_df(df_in, y_col, cycle_col='1C Cycle Excl First'):
    default_result = {
        'f_1cyc': np.nan,
        'f_50cyc': np.nan,
        'f_endm50cyc': np.nan,
        'f_endm1cyc': np.nan,
        'df_dcyc_1_50': np.nan,
        'df_dcyc_50_endm50': np.nan,
        'df_dcyc_endm50_endm1': np.nan,
        'df_dcyc_1_endm1': np.nan,
        'abruptness': np.nan,
    }

    if y_col not in df_in.columns or cycle_col not in df_in.columns:
        return default_result

    s = df_in[[cycle_col, y_col]].copy()
    s.columns = [cycle_col, 'y']
    s[cycle_col] = pd.to_numeric(s[cycle_col], errors='coerce')
    s['y'] = pd.to_numeric(s['y'], errors='coerce')
    s = s.dropna(subset=[cycle_col, 'y']).sort_values(cycle_col).drop_duplicates(subset=[cycle_col])

    if len(s) == 0:
        return default_result

    max_cycle = s[cycle_col].max()

    if pd.notna(max_cycle) and max_cycle >= 100:
        f_endm50 = get_end_minus_n_value(s, 50, cycle_col=cycle_col)
        slope_50_to_endm50 = calc_slope_if_valid(
            s, 50, max_cycle - 50,
            cycle_col=cycle_col,
            require_max_cycle_at_least=100
        )
        slope_endm50_to_endm1 = calc_slope_if_valid(
            s, max_cycle - 50, max_cycle - 1,
            cycle_col=cycle_col,
            require_max_cycle_at_least=100
        )
    else:
        f_endm50 = np.nan
        slope_50_to_endm50 = np.nan
        slope_endm50_to_endm1 = np.nan

    return {
        'f_1cyc': get_exact_cycle_value(s, 1, cycle_col=cycle_col),
        'f_50cyc': get_exact_cycle_value(s, 50, cycle_col=cycle_col),
        'f_endm50cyc': f_endm50,
        'f_endm1cyc': get_end_minus_one_value(s, cycle_col=cycle_col),
        'df_dcyc_1_50': calc_slope_between_points(s, 1, 50, cycle_col=cycle_col),
        'df_dcyc_50_endm50': slope_50_to_endm50,
        'df_dcyc_endm50_endm1': slope_endm50_to_endm1,
        'df_dcyc_1_endm1': calc_slope_1_to_endm1(s, cycle_col=cycle_col),
        'abruptness': calc_abruptness(s, cycle_col=cycle_col),
    }


def average_between_cycles(df_in, y_col, start_cycle, end_cycle, cycle_col='1C Cycle Excl First'):
    if y_col not in df_in.columns or cycle_col not in df_in.columns or len(df_in) == 0:
        return np.nan

    s = df_in[[cycle_col, y_col]].copy()
    s[cycle_col] = pd.to_numeric(s[cycle_col], errors='coerce')
    s[y_col] = pd.to_numeric(s[y_col], errors='coerce')
    s = s.dropna(subset=[cycle_col, y_col])

    s = s[(s[cycle_col] >= start_cycle) & (s[cycle_col] <= end_cycle)].copy()
    if len(s) == 0:
        return np.nan

    return s[y_col].mean()


def extract_ce_like_features(df_in, y_col, cycle_col='1C Cycle Excl First'):
    default_result = {
        'f_1cyc': np.nan,
        'avg_1_25': np.nan,
        'avg_25_100': np.nan,
        'avg_100_endm50': np.nan,
        'avg_endm50_endm1': np.nan,
    }

    if y_col not in df_in.columns or cycle_col not in df_in.columns:
        return default_result

    s = df_in[[cycle_col, y_col]].copy()
    s[cycle_col] = pd.to_numeric(s[cycle_col], errors='coerce')
    s[y_col] = pd.to_numeric(s[y_col], errors='coerce')
    s = s.dropna(subset=[cycle_col, y_col]).sort_values(cycle_col).drop_duplicates(subset=[cycle_col])

    if len(s) == 0:
        return default_result

    max_cycle = s[cycle_col].max()

    result = {
        'f_1cyc': get_exact_cycle_value(
            s.rename(columns={y_col: 'y'}),
            1,
            cycle_col=cycle_col
        ),
        'avg_1_25': average_between_cycles(s, y_col, 1, 25, cycle_col=cycle_col),
        'avg_25_100': average_between_cycles(s, y_col, 25, 100, cycle_col=cycle_col),
        'avg_100_endm50': np.nan,
        'avg_endm50_endm1': np.nan,
    }

    if pd.notna(max_cycle) and max_cycle >= 100:
        endm50 = max_cycle - 50
        endm1 = max_cycle - 1

        if endm50 >= 100:
            result['avg_100_endm50'] = average_between_cycles(s, y_col, 100, endm50, cycle_col=cycle_col)

        if endm1 >= endm50 and endm50 >= 1:
            result['avg_endm50_endm1'] = average_between_cycles(s, y_col, endm50, endm1, cycle_col=cycle_col)

    return result


# =========================
# C10 summary helpers
# =========================
def safe_pct_delta(v1, v2):
    if pd.isna(v1) or pd.isna(v2) or v1 == 0:
        return np.nan
    return (v2 - v1) / v1 * 100


def safe_rate_per_cycle(v1, v2, cyc1, cyc2):
    if pd.isna(v1) or pd.isna(v2) or pd.isna(cyc1) or pd.isna(cyc2):
        return np.nan
    if cyc2 <= cyc1:
        return np.nan
    if v1 == 0:
        return np.nan
    return ((v2 - v1) / v1 * 100) / (cyc2 - cyc1)


def build_c10_summary_row(
    cell,
    electrolyte_name,
    df_c10,
    df_kinetic,
    soc50_df
):
    row = {
        'Barcode': cell,
        'Electrolyte': electrolyte_name,
    }

    c10_cols = [
        'Charge Capacity',
        'Discharge Capacity',
        'Charge Energy (mWh)',
        'Discharge Energy (mWh)',
        'Average Charge Voltage',
        'Average Discharge Voltage',
        'Coulombic Efficiency (%)',
        'RCE',
        'Charge Resistance',
        'Discharge Resistance',
        '100% ACR',
        'Energy Efficiency %',
        'Voltage Slippage',
        'V Polarization',
    ]

    soc50_cols = ['DCIR', 'DCIR0', 'DCIRt']

    idx_map = {
        'f_1': 0,
        'f_2': 1,
        'f_end': -1,
    }

    c10_snapshots = {}
    soc50_snapshots = {}

    for snap_name, idx in idx_map.items():
        # C10
        if df_c10 is not None and len(df_c10) > 0:
            real_idx = idx if idx >= 0 else len(df_c10) - 1
            if 0 <= real_idx < len(df_c10):
                c10_row = df_c10.iloc[real_idx]
                c10_snapshots[snap_name] = c10_row

                for c in c10_cols:
                    row[f'{snap_name} | C10 | {c}'] = c10_row[c] if c in df_c10.columns else np.nan
            else:
                c10_snapshots[snap_name] = None
                for c in c10_cols:
                    row[f'{snap_name} | C10 | {c}'] = np.nan
        else:
            c10_snapshots[snap_name] = None
            for c in c10_cols:
                row[f'{snap_name} | C10 | {c}'] = np.nan

        # SOC50
        if soc50_df is not None and len(soc50_df) > 0:
            real_idx = idx if idx >= 0 else len(soc50_df) - 1
            if 0 <= real_idx < len(soc50_df):
                soc50_row = soc50_df.iloc[real_idx]
                soc50_snapshots[snap_name] = soc50_row
                for c in soc50_cols:
                    row[f'{snap_name} | SOC50 | {c}'] = soc50_row[c] if c in soc50_df.columns else np.nan
            else:
                soc50_snapshots[snap_name] = None
                for c in soc50_cols:
                    row[f'{snap_name} | SOC50 | {c}'] = np.nan
        else:
            soc50_snapshots[snap_name] = None
            for c in soc50_cols:
                row[f'{snap_name} | SOC50 | {c}'] = np.nan

    # only keep end cycle info
    if c10_snapshots['f_end'] is not None and 'Map Corrected Cycle' in c10_snapshots['f_end'].index:
        row['f_end | C10 | Map Corrected Cycle'] = c10_snapshots['f_end']['Map Corrected Cycle']
    else:
        row['f_end | C10 | Map Corrected Cycle'] = np.nan

    # delta blocks
    kinetic_1 = df_kinetic.iloc[0] if df_kinetic is not None and len(df_kinetic) >= 1 else None
    kinetic_end = df_kinetic.iloc[-1] if df_kinetic is not None and len(df_kinetic) >= 1 else None

    row['delta_1_to_2 | Kinetic Fading (%)'] = kinetic_1['Kinetic Fading (%)'] if kinetic_1 is not None and 'Kinetic Fading (%)' in kinetic_1.index else np.nan
    row['delta_1_to_end | Kinetic Fading (%)'] = kinetic_end['Kinetic Fading (%)'] if kinetic_end is not None and 'Kinetic Fading (%)' in kinetic_end.index else np.nan

    c10_f1 = c10_snapshots['f_1']
    c10_f2 = c10_snapshots['f_2']
    c10_fend = c10_snapshots['f_end']

    cyc1 = c10_f1['Map Corrected Cycle'] if c10_f1 is not None and 'Map Corrected Cycle' in c10_f1.index else np.nan
    cyc2 = c10_f2['Map Corrected Cycle'] if c10_f2 is not None and 'Map Corrected Cycle' in c10_f2.index else np.nan
    cycend = c10_fend['Map Corrected Cycle'] if c10_fend is not None and 'Map Corrected Cycle' in c10_fend.index else np.nan

    for c in c10_cols:
        v1 = c10_f1[c] if c10_f1 is not None and c in c10_f1.index else np.nan
        v2 = c10_f2[c] if c10_f2 is not None and c in c10_f2.index else np.nan
        vend = c10_fend[c] if c10_fend is not None and c in c10_fend.index else np.nan

        row[f'delta_1_to_2 | C10 | {c} | pct'] = safe_pct_delta(v1, v2)
        row[f'delta_1_to_2 | C10 | {c} | pct_per_cyc'] = safe_rate_per_cycle(v1, v2, cyc1, cyc2)

        row[f'delta_1_to_end | C10 | {c} | pct'] = safe_pct_delta(v1, vend)
        row[f'delta_1_to_end | C10 | {c} | pct_per_cyc'] = safe_rate_per_cycle(v1, vend, cyc1, cycend)

    return row


def build_full_cell_table(df, total_cap, acr_string):
    out = df.copy()
    extra_cols = [
        'Cycle',
        'Total Charge Capacity(mAh)',
        'Total Step time',
        f'{acr_string}% ACR',
        f'{acr_string}% Time',
        '20% Time',
        '20% Time Ratio'
    ]
    extra_cols = [c for c in extra_cols if c in total_cap.columns]
    extra = total_cap[extra_cols].drop_duplicates(subset=['Cycle']).copy()
    out = out.merge(extra, on='Cycle', how='left')
    return out


def build_1c_ex1_table(df_1c_ex1, total_cap_1c_ex1, acr_string):
    out = df_1c_ex1.copy()
    extra_cols = [
        'Cycle',
        '1C Cycle Excl First',
        'Total Charge Capacity(mAh)',
        'Total Step time',
        f'{acr_string}% ACR',
        f'{acr_string}% Time',
        '20% Time',
        '20% Time Ratio'
    ]
    extra_cols = [c for c in extra_cols if c in total_cap_1c_ex1.columns]
    extra = total_cap_1c_ex1[extra_cols].drop_duplicates(subset=['Cycle', '1C Cycle Excl First']).copy()
    out = out.merge(extra, on=['Cycle', '1C Cycle Excl First'], how='left')
    return out


def build_c10_combined_table(df_c10, df_kinetic, soc50_df):
    pieces = []

    if df_c10 is not None and len(df_c10) > 0:
        temp = df_c10.copy().reset_index(drop=True)
        temp.insert(0, 'Row ID', np.arange(1, len(temp) + 1))
        temp = temp.rename(columns={'Map Corrected Cycle': 'C10 | Map Corrected Cycle'})
        temp = temp.rename(columns={c: f'C10 | {c}' for c in temp.columns if c not in ['Row ID', 'C10 | Map Corrected Cycle']})
        pieces.append(temp)

    if df_kinetic is not None and len(df_kinetic) > 0:
        temp = df_kinetic.copy().reset_index(drop=True)
        temp.insert(0, 'Row ID', np.arange(1, len(temp) + 1))
        temp = temp.rename(columns={'End C/10 Map Cycle': 'Kinetic | End C/10 Map Cycle'})
        temp = temp.rename(columns={c: f'Kinetic | {c}' for c in temp.columns if c not in ['Row ID', 'Kinetic | End C/10 Map Cycle']})
        pieces.append(temp)

    if soc50_df is not None and len(soc50_df) > 0:
        temp = soc50_df.copy().reset_index(drop=True)
        temp.insert(0, 'Row ID', np.arange(1, len(temp) + 1))
        temp = temp.rename(columns={'Corrected Cycle': 'SOC50 | DCIR Cycle'})
        temp = temp.rename(columns={c: f'SOC50 | {c}' for c in temp.columns if c not in ['Row ID', 'SOC50 | DCIR Cycle']})
        pieces.append(temp)

    if len(pieces) == 0:
        return pd.DataFrame()

    out = pieces[0].copy()
    for p in pieces[1:]:
        out = out.merge(p, on='Row ID', how='outer')

    return out
