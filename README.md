## Normative diffusion models (tabular) — UKB RAP / paper release

This repository contains the code used to **train and evaluate conditional diffusion models for normative modelling** on tabular neuroimaging-derived phenotypes (IDPs), with optional **GAMLSS** baselines. It is structured to support a paper submission and to reproduce the published evaluation outputs.

### What’s in here

- **`full_eval_normative_suite.py`**: end-to-end evaluation pipeline (train → sample → metrics → plots → aggregate summaries)
- **MLP diffusion backbone**:
  - **`diffusion_models.py`**: public import module for diffusion model components (preferred)
  - **`modules_2.py`**: legacy filename (kept for backwards compatibility with earlier runs)
  - **`train_diffusion.py`**: training loop (checkpointing into per-run folders)
- **SAINT diffusion backbone**:
  - **`saint_tabular.py`**: SAINT denoiser + diffusion wrapper (data-space)
  - **`train_diffusion_saint.py`**: wrapper that plugs SAINT into the shared training loop
- **Baseline**:
  - **`gamlss_adapter.py`**: GAMLSS baseline via R (`rpy2`, `gamlss`, `gamlss.dist`)
- **Config/environment**:
  - **`config.yaml`**: default “paper-style” configuration (paths are templates)
  - **`environment.yaml`**: conda environment spec
  - **`data/`**: **empty by default** (CSV files are not distributed)

### Data availability

UK Biobank data (and any derived CSVs containing UKB participant-level values) **cannot be redistributed** in this GitHub repo. This repo therefore ships with **no CSV data**. You must generate / export your own train/holdout CSVs and organise them under `data/`.

Expected CSV schema:

- **Required columns**: `age` and a second covariate (UKB default: `sex`)
- **IDP columns**: all remaining columns are treated as IDPs (one model per IDP dimension)
- **Optional**: `eid` (will be dropped if present), optional labels (see `--label_col`)

For dimensional scaling runs, you can use the provided mapping templates:

- `data/train_csv_map.json`
- `data/holdout_csv_map.json`

Update the filenames inside them to match your local data exports (the suite expects these to point to CSVs on disk).

### Installation

Conda (recommended):

```bash
conda env create -f environment.yaml
conda activate normative_diffusion
python test_imports.py
```

Notes:

- **GPU** is recommended for the full suite.
- The **GAMLSS baseline** requires a working R installation and the R packages listed in `environment.yaml`. You can skip GAMLSS by not enabling `--run_gamlss_baseline` / `--gamlss_only`.

### UKB RAP deployment (optional)

If you want to run this on UKB RAP, a simple approach is to zip the repository and upload it:

```bash
# From the directory that contains this repo
zip -r ukb_rap_bundle.zip ukb_rap_bundle/

# Upload using your preferred RAP method (CLI or web UI)
# dx upload ukb_rap_bundle.zip
```

After extraction on RAP:

```bash
unzip ukb_rap_bundle.zip
cd ukb_rap_bundle
conda env create -f environment.yaml
conda activate normative_diffusion
python test_imports.py
```

### Running the evaluation suite

Config-driven run (recommended):

```bash
python full_eval_normative_suite.py --config config.yaml
```

Override any config key from the CLI (CLI always wins), e.g.:

```bash
python full_eval_normative_suite.py --config config.yaml \
  --run_group my_run \
  --device cuda:0 \
  --mlp_epochs 1500
```

#### Reproducing the paper run group (`124_all_both`)

The paper results directory in the original project corresponds to `run_group=124_all_both`. Assuming you have placed the required CSVs at the paths referenced by your `config.yaml`, you can reproduce the same directory layout via:

```bash
python full_eval_normative_suite.py --config config.yaml --run_group 124_all_both
```

Tip: if you already have checkpoints in that output folder and just want to re-run evaluation without retraining, use `--skip_if_exists`.

#### Where outputs are written (`--results_dir`)

This release keeps the historical output layout used during development on RAP:

- `--results_dir` specifies the directory that will contain `results_full_eval/`
- By default, `--results_dir` points to the **parent** of this repo directory (matching the original `ukb_rap/` layout)

If you want outputs *inside* this repo folder instead, run with:

```bash
python full_eval_normative_suite.py --config config.yaml --results_dir .
```

### Output structure

Runs are written to:

```
<results_dir>/results_full_eval/<run_group>/
```

Key files/directories:

- `cli_args.json`: full resolved configuration for the run (incl. defaults + overrides)
- `summary.json`: aggregated records across all runs/conditions
- `ukb/`, `synth/`: per-dataset results (if a synthetic dataset is provided)
- within each dataset:
  - `mlp/`, `saint/`, `gamlss/`: per-method runs
  - `combined/`: cross-method overlays
  - `base_full/`, `learning_curves/`, `dim_scaling/`: experimental suites

Each leaf run directory contains:

- `models/`: saved checkpoints
- `results/`: JSON metrics + plots (PIT, coverage curves, centile pages, joint analyses, …)

### Quick sanity check

After installing dependencies, you can verify imports with:

```bash
python test_imports.py
```

### Citation

If you use this code, please cite the associated paper:

> (Add citation here.)
