#!/usr/bin/env bash
set -euo pipefail

# Example run script.
#
# The config file is the authoritative place for the “paper-style” defaults.
# This script shows the recommended invocation pattern (config + optional overrides).
#
# IMPORTANT: This repository does not ship with UKB/synthetic CSV data. You must
# place your local CSVs under `data/` (or adjust paths in `config.yaml`).

# Basic config-driven run (writes to the historical default results location):
python full_eval_normative_suite.py --config config.yaml --run_group example_run

# If you want outputs inside the current directory instead:
# python full_eval_normative_suite.py --config config.yaml --run_group example_run --results_dir .

