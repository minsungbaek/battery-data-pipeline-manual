"""
Intermediate dataframe I/O helpers.

The pipeline uses parquet when an engine is available and falls back to pickle
when it is not. This keeps repeated notebook/script runs from re-parsing large
Excel workbooks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _detect_parquet_engine() -> str | None:
    try:
        import pyarrow  # noqa: F401

        return "pyarrow"
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401

        return "fastparquet"
    except Exception:
        pass
    return None


_PARQUET_ENGINE = _detect_parquet_engine()


def io_format() -> str:
    """Return the active intermediate format: ``parquet`` or ``pickle``."""
    return "parquet" if _PARQUET_ENGINE is not None else "pickle"


def save_intermediate(df: pd.DataFrame, out_dir: str | Path, name: str) -> str:
    """Save a dataframe and return the absolute path written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _PARQUET_ENGINE is not None:
        path = out_dir / f"{name}.parquet"
        _coerce_for_parquet(df).to_parquet(path, engine=_PARQUET_ENGINE, index=False)
    else:
        path = out_dir / f"{name}.pkl"
        df.to_pickle(path)
    return str(path.resolve())


def load_intermediate(in_dir: str | Path, name: str) -> pd.DataFrame:
    """Load an intermediate dataframe by stem name."""
    in_dir = Path(in_dir)
    parquet_path = in_dir / f"{name}.parquet"
    pickle_path = in_dir / f"{name}.pkl"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if pickle_path.exists():
        return pd.read_pickle(pickle_path)
    raise FileNotFoundError(
        f"No intermediate file {name!r} in {in_dir}. "
        f"Available: {list_intermediate(in_dir)}"
    )


def list_intermediate(in_dir: str | Path) -> dict[str, str]:
    """Return ``{stem: file_path}`` for parquet/pickle files in a folder."""
    in_dir = Path(in_dir)
    if not in_dir.exists():
        return {}
    out = {}
    for p in sorted(in_dir.iterdir()):
        if p.suffix in (".parquet", ".pkl"):
            out[p.stem] = str(p)
    return out


def _coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce mixed object columns to parquet-friendly values."""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for c in out.columns:
        if out[c].dtype != object:
            continue
        numeric = pd.to_numeric(out[c], errors="coerce")
        if numeric.notna().sum() == out[c].notna().sum():
            out[c] = numeric
        else:
            out[c] = out[c].astype(str).where(out[c].notna(), None)
    return out


def xlsx_to_intermediate_dir(
    xlsx_path: str | Path,
    out_dir: str | Path,
    include_per_cell_full: bool = False,
    verbose: bool = True,
) -> tuple[Path, dict[str, str]]:
    """Convert a multi-sheet cycling workbook into parquet/pickle tables."""
    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Opening workbook once: {xlsx_path}")
    xl = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names
    saved: dict[str, str] = {}

    for sheet_name in ("summary_features_ex1", "summary_features_c10"):
        if sheet_name not in sheets:
            continue
        df = xl.parse(sheet_name)
        path = save_intermediate(df, out_dir, sheet_name)
        saved[sheet_name] = path
        if verbose:
            print(f"  saved {sheet_name}: {len(df)} rows -> {Path(path).name}")

    known_barcodes = []
    if "summary_features_c10" in sheets:
        try:
            sc10 = xl.parse("summary_features_c10", usecols=["Barcode"])
            known_barcodes = sc10["Barcode"].dropna().astype(str).unique().tolist()
        except Exception:
            known_barcodes = []

    def match_sheet(barcode: str, suffix: str) -> str | None:
        candidate = f"{barcode}{suffix}"
        if candidate in sheets:
            return candidate
        truncated = candidate[:31]
        if truncated in sheets:
            return truncated
        for sheet in sheets:
            if sheet.endswith(suffix) and sheet.startswith(barcode[: max(1, 31 - len(suffix))]):
                return sheet
        return None

    for output_name, suffix in (("c10_traces", "_C10"), ("ex1_traces", "_1C_ExclFirst")):
        rows = []
        for barcode in known_barcodes:
            sheet = match_sheet(barcode, suffix)
            if sheet is None:
                continue
            df = xl.parse(sheet)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "Barcode", barcode)
            rows.append(df)
        if rows:
            long_df = pd.concat(rows, ignore_index=True, sort=False)
            path = save_intermediate(long_df, out_dir, output_name)
            saved[output_name] = path
            if verbose:
                print(f"  saved {output_name}: {len(long_df)} rows -> {Path(path).name}")

    if include_per_cell_full:
        rows = []
        for barcode in known_barcodes:
            sheet = barcode[:31]
            if sheet not in sheets:
                continue
            df = xl.parse(sheet)
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, "Barcode", barcode)
            rows.append(df)
        if rows:
            full_df = pd.concat(rows, ignore_index=True, sort=False)
            path = save_intermediate(full_df, out_dir, "per_cell_full_traces")
            saved["per_cell_full_traces"] = path

    meta = {
        "source_xlsx": str(xlsx_path.resolve()),
        "parquet_engine": _PARQUET_ENGINE,
        "io_format": io_format(),
        "saved_files": saved,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    with open(out_dir / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return out_dir.resolve(), saved


if __name__ == "__main__":
    import shutil
    import tempfile

    df = pd.DataFrame({"Barcode": ["R1", "R2"], "Electrolyte": ["A", "B"], "value": [1.0, 2.0]})
    tmp = Path(tempfile.mkdtemp())
    try:
        path = save_intermediate(df, tmp, "test")
        loaded = load_intermediate(tmp, "test")
        assert loaded.shape == df.shape
        print(f"I/O round trip OK via {io_format()}: {path}")
    finally:
        shutil.rmtree(tmp)
