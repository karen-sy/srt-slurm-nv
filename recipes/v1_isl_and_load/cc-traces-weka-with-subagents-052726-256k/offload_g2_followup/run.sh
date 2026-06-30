#!/usr/bin/env bash
set -euo pipefail

ROOT=/localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/offload_g2_followup

# Required first pass from offload_g2_probe_recap.md.
srtctl apply -f "$ROOT/01_agg_offload_nogate_400g_12gpu_tep4x3.yaml"
srtctl apply -f "$ROOT/02_agg_offload_nogate_120g_logs_12gpu_tep4x3.yaml"

# Optional follow-ups after P1/P2:
# srtctl apply -f "$ROOT/03_agg_offload_threshold2_400g_12gpu_tep4x3.yaml"
# srtctl apply -f "$ROOT/04_disagg_tuned_offload_400g_14gpu_tep4x2p_tep2x3d.yaml"

