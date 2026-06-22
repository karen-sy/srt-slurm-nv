#!/usr/bin/env bash
set -euo pipefail

ROOT=/localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/condp_sticky_offload

srtctl apply -f "$ROOT/01_decode_offload_lru_14gpu_tep4x2p_tep2x3d.yaml"
srtctl apply -f "$ROOT/02_decode_offload_arc_14gpu_tep4x2p_tep2x3d.yaml"
srtctl apply -f "$ROOT/03_decode_offload_lru_block256_14gpu_tep4x2p_tep2x3d.yaml"
