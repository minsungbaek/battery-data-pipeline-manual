"""
feature_transformations.py
==========================

Step 1 of the transferable-feature pipeline:
    (1a) feature-class-aware linearizing transformation
    (1b) scale-aware robust z-score standardization

The result is each feature on a "deviation in robust-sigma units" scale, which
makes "1 % change in CE" comparable in magnitude to "10 % change in capacity"
because the same z-magnitude reflects the same *population significance*.

Robust statistics (median + IQR) are used instead of mean + std because:
    - small N (~50-500 cells) is sensitive to outliers in std
    - many battery features have skewed / heavy-tailed distributions
    - IQR-based standardization is monotone and preserves rank-order

This module operates on per-cell **summary** features (one row per cell), not
on per-cycle traces.  It auto-detects the feature class from the column name.

Maintainer: Minsung Baek
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# (A) Feature-class detection
# ---------------------------------------------------------------------------

# Each class -> (uses_linearize, default_transform_name)
FEATURE_CLASSES = [
    'abruptness',         # shape, dimensionless
    'kinetic_fading',     # already %, just standardize
    'time_ratio',         # bounded [0,1]
    'cum_loss',           # cumulative % loss (linear)
    'cycle_life',         # % retention (already normalized)
    'rate_pct_per_cyc',   # rate %/cycle
    'delta_pct',          # already a percentage delta
    'slope',              # rate in raw units/cycle
    'ce_rce',             # bounded near 100 - use HI1
    'energy_efficiency',  # bounded near 90 - log(1 - x/100)
    'resistance',         # log-normal: DCIR, ACR, R_ch, R_disch
    'voltage_small',      # small drift in V - V_slip, V_pol
    'voltage_abs',        # absolute mean voltage
    'capacity_or_energy', # raw mAh or mWh
    'default',            # unknown
]


def classify_feature(name: str) -> str:
    """
    Heuristic: classify a summary-feature column name into one of FEATURE_CLASSES.
    Order is *most specific first* (so 'cycle_life_*' matches before generic 'cycle').
    """
    n = str(name).lower().replace(' ', '_').replace('|', '_').replace('__', '_')

    if 'abruptness' in n:
        return 'abruptness'
    if 'kinetic_fading' in n or 'kinetic_fade' in n:
        return 'kinetic_fading'
    if 'time_ratio' in n:
        return 'time_ratio'
    if 'cum_ce_loss' in n or 'cum_rce_loss' in n:
        return 'cum_loss'
    if 'cycle_life' in n:
        return 'cycle_life'
    if 'pct_per_cyc' in n:
        return 'rate_pct_per_cyc'
    if '_pct' in n and 'delta' in n:
        return 'delta_pct'
    if 'df_dcyc' in n or 'df_d_cyc' in n:
        return 'slope'
    # CE / RCE - careful, must NOT match 'cycle_life' or 'cum_*ce*loss'
    if (re.search(r'(^|_)ce_', n) or re.search(r'_ce(_|$)', n) or
        'rce' in n or 'coulombic_efficiency' in n):
        return 'ce_rce'
    if 'energy_efficiency' in n:
        return 'energy_efficiency'
    if 'dcir' in n or 'resistance' in n or 'acr' in n:
        return 'resistance'
    if ('v_polarization' in n or 'voltage_slippage' in n or
        'v_slip' in n or 'voltage_polarization' in n):
        return 'voltage_small'
    if 'avg_charge_voltage' in n or 'avg_discharge_voltage' in n or 'average_charge_voltage' in n or 'average_discharge_voltage' in n:
        return 'voltage_abs'
    if 'capacity' in n or 'energy' in n:
        return 'capacity_or_energy'
    return 'default'


# ---------------------------------------------------------------------------
# (B) Linearizing transformations
# ---------------------------------------------------------------------------

def hi1_transform(x):
    """
    HI1 = log(exp(x/100) - 1).
    For x near 100, spreads tiny differences (CE 99.99 vs 99.95 -> distinguishable).
    For x = 100 exactly, blows up; clip just below.
    """
    x = pd.to_numeric(x, errors='coerce')
    x = np.where(np.isfinite(x), x, np.nan)
    # clip at 100 - epsilon to avoid log(0) when CE = 100% exactly
    x_safe = np.clip(x / 100.0, -50, 50)
    val = np.exp(x_safe) - 1.0
    val = np.where(val > 0, val, np.nan)
    return np.log(np.where(val > 0, val, np.nan))


def log_one_minus_eff(x):
    """
    For energy efficiency near 100 %: log(1 - x/100) -> spreads inefficiency.
    """
    x = pd.to_numeric(x, errors='coerce')
    val = 1.0 - np.clip(x / 100.0, -np.inf, 1.0 - 1e-9)
    val = np.where(val > 0, val, np.nan)
    return np.log(val)


def safe_log_abs(x, eps=1e-12):
    """log(|x|) with safe handling of zeros."""
    x = pd.to_numeric(x, errors='coerce')
    a = np.abs(x)
    a = np.where(a > eps, a, np.nan)
    return np.log(a)


# Map class -> linearizing function (or None for identity)
LINEARIZE_FN = {
    'ce_rce':            hi1_transform,
    'energy_efficiency': log_one_minus_eff,
    'resistance':        safe_log_abs,
    'voltage_small':     None,           # small drift, identity is fine
    'voltage_abs':       None,
    'capacity_or_energy': None,
    'cum_loss':          None,
    'cycle_life':        None,
    'time_ratio':        None,
    'kinetic_fading':    None,
    'delta_pct':         None,
    'rate_pct_per_cyc':  None,
    'slope':             None,
    'abruptness':        None,
    'default':           None,
}


def linearize(s: pd.Series, cls: str) -> pd.Series:
    fn = LINEARIZE_FN.get(cls, None)
    if fn is None:
        return pd.to_numeric(s, errors='coerce')
    return pd.Series(fn(s), index=s.index)


# ---------------------------------------------------------------------------
# (C) Robust scale-aware standardization
# ---------------------------------------------------------------------------

def robust_zscore(s: pd.Series, center: str = 'median', scale: str = 'iqr',
                  iqr_to_sigma: bool = True) -> pd.Series:
    """
    Standardize so the result is comparable across heterogeneous features.

    - center='median' (default) for robustness; 'mean' for classic z-score
    - scale='iqr' (default) uses interquartile range; 'std' uses population std
    - iqr_to_sigma=True scales IQR by 1/1.349 so result matches Gaussian sigma units

    Returns a Series of "robust standard deviations from median".  A value of
    +2 means "2 robust-sigmas above the population median".  For features with
    near-zero variance, returns NaN.
    """
    s = pd.to_numeric(s, errors='coerce')

    if center == 'median':
        c = s.median()
    else:
        c = s.mean()

    if scale == 'iqr':
        sc = s.quantile(0.75) - s.quantile(0.25)
        if iqr_to_sigma:
            sc = sc / 1.349  # Gaussian: IQR ~ 1.349 sigma
    else:
        sc = s.std()

    if pd.isna(sc) or sc == 0:
        return pd.Series(np.nan, index=s.index)

    return (s - c) / sc


# ---------------------------------------------------------------------------
# (C-bis) Default exclusion patterns (target / metadata / known-broken columns)
# ---------------------------------------------------------------------------
#
# These columns should NEVER enter the standardized feature matrix because they
# are either (a) the prediction target, (b) metadata / cycle indices, or
# (c) known-broken measurements that produce non-physical values.
#
# Match is by exact column name OR by regex pattern.

DEFAULT_EXCLUDE_EXACT = {
    'Final 1C Cycle Number',          # = lifetime, the prediction TARGET (would leak)
    'Final 1C Cycle Number - 50',
    'f_end | C10 | Map Corrected Cycle',  # = cycle number when test ended (= lifetime proxy)
    'f_1 | C10 | Energy Efficiency %',    # systematic >100% (Si formation artifact, non-physical)
    'f_1 | C10 | RCE',                    # first cycle RCE undefined (no previous discharge)
}

DEFAULT_EXCLUDE_REGEX = [
    # add patterns here as needed, e.g. r'.*_first_cycle_.*' to exclude all first-cycle features
]


def is_excluded(name: str,
                exclude_exact=DEFAULT_EXCLUDE_EXACT,
                exclude_regex=DEFAULT_EXCLUDE_REGEX) -> bool:
    if name in exclude_exact:
        return True
    for pat in exclude_regex:
        if re.search(pat, name):
            return True
    return False


# ---------------------------------------------------------------------------
# (C-ter) Bad-cell detection (extreme-z voting)
# ---------------------------------------------------------------------------

def flag_bad_cells(transformed_df: pd.DataFrame,
                   id_col: str = 'Barcode',
                   z_threshold: float = 5.0,
                   extreme_count_threshold: int = 10):
    """
    Returns
    -------
    flag_df : DataFrame[id_col, extreme_count, is_bad_cell]
        A 'bad cell' is one whose features produce |z|>z_threshold across
        many (>extreme_count_threshold) features.

    These are typically cells with manufacturing defects, instrumentation
    glitches, or runaway side reactions; they distort the population statistics
    and should be removed before downstream modelling.
    """
    if id_col not in transformed_df.columns:
        return pd.DataFrame(columns=[id_col, 'extreme_count', 'is_bad_cell'])

    feat_cols = [c for c in transformed_df.columns
                 if c != id_col and pd.api.types.is_numeric_dtype(transformed_df[c])]

    counts = []
    for _, row in transformed_df.iterrows():
        z = pd.to_numeric(row[feat_cols], errors='coerce').abs()
        counts.append(int((z > z_threshold).sum()))

    out = pd.DataFrame({
        id_col: transformed_df[id_col].values,
        'extreme_count': counts,
    })
    out['is_bad_cell'] = out['extreme_count'] > extreme_count_threshold
    return out


# ---------------------------------------------------------------------------
# (D) FeatureScaler - sklearn-style fit/transform with saveable parameters
# ---------------------------------------------------------------------------

class FeatureScaler:
    """
    Per-feature linearizing transform + robust z-score with FROZEN scale.

    Why frozen?  If we recompute the IQR on every new DOE, then '+2sigma' on this
    DOE means something different from '+2sigma' on the next DOE.  The whole point
    is to be able to *accumulate* normalized data across DOEs and have the
    sigma-unit mean the same physical magnitude.

    Use:
        scaler = FeatureScaler()
        scaler.fit(reference_df)          # compute centers + scales once
        scaler.save('feature_scales.yaml')

        # later, on a new DOE:
        scaler2 = FeatureScaler.load('feature_scales.yaml')
        new_df_z = scaler2.transform(new_df)

    Center options
    --------------
        'global_median' : center on this dataset's per-feature median (default).
                          Suitable when comparing absolute features.
        'zero'          : leave centered at 0; appropriate when the input is
                          already a delta (Step 3 output).
        'baseline_per_doe' : center is recomputed per DOE using a marked
                          baseline electrolyte (handled in apply, not here).

    Scale is always 'robust_iqr' (population IQR / 1.349, locked at fit time).
    """

    def __init__(self, center: str = 'global_median'):
        assert center in ('global_median', 'zero', 'baseline_per_doe')
        self.center_mode = center
        self.params: dict = {}        # {feature: {center, scale, class, linearize_fn_name}}
        self.fitted: bool = False

    # ----- fit ----------------------------------------------------------------

    def fit(self, df: pd.DataFrame,
            id_cols=('Barcode', 'Electrolyte'),
            min_valid: int = 4,
            exclude_targets: bool = True):
        """
        Fit per-feature center+scale parameters from the reference dataset.

        Parameters
        ----------
        exclude_targets : bool, default True
            When True, any column matching DEFAULT_EXCLUDE_EXACT/REGEX is
            skipped (e.g., 'Final 1C Cycle Number' is the lifetime *target*,
            not a feature; including it would leak the answer).
        """
        self.params = {}
        self.skipped: dict = {}    # {feature: reason}
        for c in df.columns:
            if c in id_cols:
                continue
            if exclude_targets and is_excluded(c):
                self.skipped[c] = 'excluded_target_or_broken'
                continue
            s_raw = pd.to_numeric(df[c], errors='coerce')
            n_valid = int(s_raw.notna().sum())
            if n_valid < min_valid or s_raw.std() == 0:
                continue

            cls = classify_feature(c)
            s_lin = linearize(s_raw, cls)
            if s_lin.notna().sum() < min_valid or (s_lin.std() or 0) == 0:
                # linearization broke; fall back to raw
                s_lin = s_raw
                cls_used = cls + '__raw_fallback'
            else:
                cls_used = cls

            iqr = s_lin.quantile(0.75) - s_lin.quantile(0.25)
            scale = float(iqr / 1.349) if (pd.notna(iqr) and iqr > 0) else float(s_lin.std() or 1.0)
            if scale == 0 or not np.isfinite(scale):
                scale = 1.0

            if self.center_mode == 'global_median':
                center = float(s_lin.median())
            elif self.center_mode == 'zero':
                center = 0.0
            else:
                center = float(s_lin.median())  # placeholder; per-DOE handled in transform

            self.params[c] = {
                'class': cls_used,
                'center': center,
                'scale': scale,
                'pre_median_raw': float(s_raw.median()),
                'pre_iqr_raw':    float(s_raw.quantile(0.75) - s_raw.quantile(0.25)),
                'pre_std_raw':    float(s_raw.std()),
                'pre_min_raw':    float(s_raw.min()),
                'pre_max_raw':    float(s_raw.max()),
                'n_valid_at_fit': n_valid,
            }
        self.fitted = True
        return self

    # ----- transform ----------------------------------------------------------

    def transform(self, df: pd.DataFrame, id_cols=('Barcode', 'Electrolyte')) -> pd.DataFrame:
        assert self.fitted, "Call .fit() first or .load() saved params."
        out = df.copy()
        for c in df.columns:
            if c in id_cols:
                continue
            if c not in self.params:
                # feature not in saved scaler - leave it untouched but flag it
                continue
            p = self.params[c]
            s_raw = pd.to_numeric(df[c], errors='coerce')
            cls_clean = p['class'].replace('__raw_fallback', '')
            s_lin = linearize(s_raw, cls_clean) if '__raw_fallback' not in p['class'] else s_raw
            out[c] = (s_lin - p['center']) / p['scale']
        return out

    def fit_transform(self, df, id_cols=('Barcode', 'Electrolyte'), exclude_targets=True):
        self.fit(df, id_cols=id_cols, exclude_targets=exclude_targets)
        return self.transform(df, id_cols=id_cols)

    # ----- persistence --------------------------------------------------------

    def save(self, path: str):
        import json
        meta = {
            'center_mode': self.center_mode,
            'params': self.params,
            'skipped': getattr(self, 'skipped', {}),
            'created_at': pd.Timestamp.now().isoformat(),
            'n_features': len(self.params),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        obj = cls(center=meta.get('center_mode', 'global_median'))
        obj.params = meta['params']
        obj.skipped = meta.get('skipped', {})
        obj.fitted = True
        return obj

    def to_log_dataframe(self) -> pd.DataFrame:
        rows = []
        for feat, p in self.params.items():
            rows.append({'feature': feat, **p})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (D-bis) One-shot convenience
# ---------------------------------------------------------------------------

def apply_step1(df, id_cols=('Barcode', 'Electrolyte'), return_log=True,
                min_valid=4, exclude_targets=True):
    sc = FeatureScaler(center='global_median').fit(df, id_cols=id_cols,
                                                    min_valid=min_valid,
                                                    exclude_targets=exclude_targets)
    out = sc.transform(df, id_cols=id_cols)
    log_df = sc.to_log_dataframe()
    return (out, log_df) if return_log else out


# ---------------------------------------------------------------------------
# (E) Self-test
# ---------------------------------------------------------------------------

def _self_test():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        'Barcode':     [f'B{i:03d}' for i in range(n)],
        'Electrolyte': ['baseline']*25 + ['additive']*25,
        'discharge_capacity_ex1_f_endm1cyc': rng.normal(600, 5, n),
        'ce_ex1_avg_endm50_endm1':           99.85 + rng.normal(0, 0.01, n),
        'f_end | SOC50 | DCIR':              np.exp(rng.normal(np.log(0.23), 0.05, n)),
        'v_slippage_ex1_f_endm1cyc':         3.625 + rng.normal(0, 0.002, n),
        'v_polarization_ex1_f_endm1cyc':     0.478 + rng.normal(0, 0.017, n),
        # Should be excluded automatically:
        'Final 1C Cycle Number':             rng.uniform(380, 440, n),
    })

    sc = FeatureScaler()
    out = sc.fit_transform(df)
    log = sc.to_log_dataframe()

    assert 'Final 1C Cycle Number' not in sc.params, "exclude failed"
    assert 'Final 1C Cycle Number' in sc.skipped, "should be in skipped"

    for feat in log['feature']:
        z = pd.to_numeric(out[feat], errors='coerce').dropna()
        med = z.median()
        iqr = z.quantile(0.75) - z.quantile(0.25)
        assert abs(med) < 1e-6, f"{feat} median != 0: {med}"
        assert abs(iqr - 1.349) < 0.02, f"{feat} IQR != 1.349: {iqr}"

    # Inject 1 bad cell (extreme outliers everywhere)
    df_bad = df.copy()
    df_bad.iloc[3, 2:] = df_bad.iloc[3, 2:].astype(float) * 5.0  # 5x off
    out_bad = sc.transform(df_bad)
    flag = flag_bad_cells(out_bad, id_col='Barcode', z_threshold=5.0,
                          extreme_count_threshold=2)
    assert flag['is_bad_cell'].sum() >= 1, "did not flag injected bad cell"

    print("[self_test] FeatureScaler fit/transform with exclude   OK")
    print("[self_test] flag_bad_cells detected injected outlier   OK")

    sc.save('/tmp/test_scales.json')
    sc2 = FeatureScaler.load('/tmp/test_scales.json')
    out2 = sc2.transform(df)
    assert np.allclose(out.select_dtypes('number').values,
                       out2.select_dtypes('number').values, equal_nan=True)
    print("[self_test] save/load round-trip                        OK")


if __name__ == '__main__':
    _self_test()
