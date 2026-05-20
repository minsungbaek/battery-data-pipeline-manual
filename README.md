# Battery Data Pipeline

Manual notebooks and helper modules for turning battery test data into per-cell
feature tables and per-DOE summary workbooks.

This repository is organized around a manual workflow: process each raw test
type first, then run the Stage 1 + Stage 2 summary notebook.

## Repository Contents

| File | Purpose |
| --- | --- |
| `Cycling_Processor.ipynb` | Manual cycling NDA processing into Excel and `.stage1ready.pkl`. |
| `HT_Storage_Processor.ipynb` | Manual high-temperature storage processing into Excel and `.stage1ready.pkl`. |
| `Rate_Capability_Processor.ipynb` | Manual rate-capability processing into Excel and `.stage1ready.pkl`. |
| `stage1_2_pipeline.ipynb` | Manual Stage 1 extraction and Stage 2 DOE summary workflow. |
| `stage1_extractor.py` | Extracts per-cell Stage 1 feature tables. |
| `stage2_aggregator.py` | Builds lot-level Stage 2 summaries and Excel outputs. |
| `feature_transformations.py` | Robust feature scaling and bad-cell flagging. |
| `lli_lam_io.py` | Parquet-first, pickle-fallback intermediate table I/O. |
| `selected_features.yaml` | Optional curated feature lists for notebooks and audits. |
| `config.example.json` | Example local configuration. |
| `requirements.txt` | Python dependencies. |

## Manual Workflow

### 1. Cycling

Open `Cycling_Processor.ipynb` and run it for each cycling test item, such as
RT 1C1C, HT 1C1C, HT 1_5C3C, or RT 4C1C.

Expected outputs:

```text
single_test_feature_analysis_*.xlsx
*.stage1ready.pkl
```

### 2. HT Storage

Open `HT_Storage_Processor.ipynb` and provide the storage raw-data folders,
metadata workbook, and optional thickness/ACR workbook.

Expected outputs:

```text
HT_storage_output.xlsx
HT_storage_output.stage1ready.pkl
```

### 3. Rate Capability

Open `Rate_Capability_Processor.ipynb` and provide the rate-capability raw NDA
folder, metadata workbook, and optional cycle map.

Expected outputs:

```text
rate_cap_output.xlsx
rate_cap_output.stage1ready.pkl
```

### 4. Stage 1 + Stage 2 Summary

Open `stage1_2_pipeline.ipynb` and manually select each test item source
workbook. Then run Stage 2 to generate the DOE-level summary workbook and
downstream table.

Expected outputs:

```text
stage1_<test_item>/summary_features.parquet
stage2_<DOE>.xlsx
stage2_<DOE>.parquet
```

## Setup

```bash
python -m pip install -r requirements.txt
```

For parquet output, `pyarrow` is recommended. If parquet support is unavailable,
the pipeline automatically falls back to pickle.

## Public Data Policy

Do not commit raw cell data, generated workbooks, metadata sheets, `.pkl`,
`.parquet`, scaler JSONs, or real DOE folders unless they are intentionally
sanitized examples. The included `.gitignore` blocks those files by default.
