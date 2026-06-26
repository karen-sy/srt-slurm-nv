#!/usr/bin/env bash
set -euo pipefail

ROOT=/localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/condp_sticky_bidi

srtctl apply -f "$ROOT/02_nooffload_bidi_16gpu_tep4x2p_tep4x2d_decode_gate.yaml"
srtctl apply -f "$ROOT/03_nooffload_bidi_12gpu_tep4x1p_tep4x2d_prefill_scarce.yaml"
srtctl apply -f "$ROOT/01_nooffload_bidi_14gpu_tep4x2p_tep2x3d_decode_gate.yaml"
srtctl apply -f "$ROOT/05_offload_bidi_lru_14gpu_tep4x2p_tep2x3d.yaml"
