#!/usr/bin/env bash
set -euo pipefail

ROOT=/localhome/local-karenc/srt-slurm/recipes/deepseek-v4-pro/pure_disagg

srtctl apply -f "$ROOT/01_vllm_dsv4_ctx2dep8_gen1dep16_c224_c384.yaml"
srtctl apply -f "$ROOT/02_vllm_dsv4_ctx3dep8_gen1dep16_c576.yaml"
srtctl apply -f "$ROOT/03_vllm_dsv4_ctx6dep4_gen1dep16_c768.yaml"
srtctl apply -f "$ROOT/04_vllm_dsv4_ctx3dep4_gen1dep8_c512.yaml"
