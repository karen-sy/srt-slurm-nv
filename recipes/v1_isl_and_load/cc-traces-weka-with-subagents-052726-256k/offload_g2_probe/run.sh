#!/usr/bin/env bash
set -euo pipefail

ROOT=/localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/offload_g2_probe

srtctl apply -f "$ROOT/01_agg_single_offload_nogate_12gpu_tep4x3.yaml"
srtctl apply -f "$ROOT/02_disagg_offload_first_nogate_14gpu_tep4x2p_tep2x3d.yaml"
