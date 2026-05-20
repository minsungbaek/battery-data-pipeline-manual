# Battery Data Pipeline

Python tools for turning battery test data into reusable per-cell feature tables
and per-DOE summary workbooks.

The workflow supports cycling, high-temperature storage, formation, and discharge
rate-capability data. Stage 0 creates fast `.stage1ready.pkl` caches from raw
exports where needed. Stage 1 extracts standardized per-cell features. Stage 2
aggregates cells to lot-level summaries, computes deltas against a baseline lot,
flags bad cells, and writes Excel plus parquet/pickle outputs.

## Repository Contents

| File | Purpose |
| --- | --- |
| `Single Point DCIR Chart newtry5float_3.ipynb` | Manual cycling NDA processing into Excel and `.stage1ready.pkl`. |
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

Open `Single Point DCIR Chart newtry5float_3.ipynb` and run it for each cycling test item, such as RT 1C1C, HT 1C1C, HT 1_5C3C, or RT 4C1C.

Expected outputs:

```text
single_test_feature_analysis_*.xlsx
*.stage1ready.pkl
```

### 2. HT Storage

Open `HT_Storage_Processor.ipynb` and provide the storage raw-data folders, metadata workbook, and optional thickness/ACR workbook.

Expected outputs:

```text
HT_storage_output.xlsx
HT_storage_output.stage1ready.pkl
```

### 3. Rate Capability

Open `Rate_Capability_Processor.ipynb` and provide the rate-capability raw NDA folder, metadata workbook, and optional cycle map.

Expected outputs:

```text
rate_cap_output.xlsx
rate_cap_output.stage1ready.pkl
```

### 4. Stage 1 + Stage 2 Summary

Open `stage1_2_pipeline.ipynb` and manually select each test item source workbook. Then run Stage 2 to generate the DOE-level summary workbook and downstream table.

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

For parquet output, `pyarrow` is recommended. If parquet support is unavailable, the pipeline automatically falls back to pickle.

## Public Data Policy

Do not commit raw cell data, generated workbooks, metadata sheets, `.pkl`, `.parquet`, scaler JSONs, or real DOE folders unless they are intentionally sanitized examples. The included `.gitignore` blocks those files by default.
