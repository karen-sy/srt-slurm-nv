#!/usr/bin/env bash
set -euo pipefail

ROOT=/localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/condp_sticky

srtctl apply -f "$ROOT/01_nooffload_14gpu_tep4x2p_tep2x3d_affinity_isolate_c96.yaml"
srtctl apply -f "$ROOT/02_nooffload_14gpu_tep4x2p_tep2x3d_decode_gate_c96.yaml"
srtctl apply -f "$ROOT/03_nooffload_20gpu_tep4x2p_tep4x3d_affinity_contrast_c96.yaml"
srtctl apply -f "$ROOT/04_nooffload_14gpu_tep4x2p_tep2x3d_concurrency_shape_decode_gate.yaml"
