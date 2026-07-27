#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Modeling perf controlled trace benchmark using aiperf.
#
# Usage:
#   bench.sh ENDPOINT MODEL_NAME TRACE_DIR K CONCURRENCIES \
#     [TTFT_THRESHOLD] [ITL_THRESHOLD] [TOKENIZER_PATH] [EXTRA_AIPERF_ARGS...]
#
# TRACE_DIR_OR_FILE may be either:
# - a P0 directory containing profile-k${K}-c${C}.jsonl files, whose first
#   C + 1 rows prime the background and injection lineages; or
# - an Experiment 2 canonical real-CD JSONL, whose first C rows prime stable
#   base lineages.
#
# Warmup rows are excluded from stats. The measured rows intentionally have no
# timestamp/delay fields; aiperf runs closed-loop by concurrency.

set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"
LIB_DIR="${SCRIPT_DIR}/../lib"

if [[ "${PROFILING_BACKEND:-}" == "trtllm" ]]; then
    # shellcheck source=../lib/profiling_trtllm.sh
    source "${LIB_DIR}/profiling_trtllm.sh"
else
    # shellcheck source=../lib/profiling.sh
    source "${LIB_DIR}/profiling.sh"
fi
profiling_init_from_env

cleanup() { stop_all_profiling; }
trap cleanup EXIT

export PYTHONUNBUFFERED=1

ENDPOINT=$1
MODEL_NAME=${2:-"test-model"}
TRACE_INPUT=$3
K=$4
CONCURRENCIES=${5:-"16,32,64"}
TTFT_THRESHOLD=${6:-2000}
ITL_THRESHOLD=${7:-25}
TOKENIZER_PATH=${8:-"/model"}
shift 8 2>/dev/null || true
EXTRA_ARGS=("$@")

ISL_BLOCK_SIZE="${AIPERF_ISL_BLOCK_SIZE:-64}"
EXTRA_INJECTION_SLOTS="${MODELING_ITL_PERF_EXTRA_INJECTION_SLOTS:-1}"
RUN_CADENCE_RAMP="${MODELING_ITL_PERF_RUN_CADENCE_RAMP:-0}"

SERVER_METRICS_ARGS=()
if [ -n "${AIPERF_SERVER_METRICS_URLS:-}" ]; then
    IFS=',' read -r -a server_metrics_urls <<< "${AIPERF_SERVER_METRICS_URLS}"
    if [ ${#server_metrics_urls[@]} -gt 0 ]; then
        SERVER_METRICS_ARGS+=(--server-metrics "${server_metrics_urls[@]}")
        SERVER_METRICS_ARGS+=(--server-metrics-formats json jsonl)
    fi
fi

BASE_DIR="${BASE_DIR:-/logs}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${BASE_DIR}/artifacts}"
mkdir -p "${ARTIFACT_DIR}"

ulimit -n 600000 2>/dev/null || ulimit -n 65536 2>/dev/null || true
export AIPERF_HTTP_SO_RCVTIMEO="${AIPERF_HTTP_SO_RCVTIMEO:-120}"

echo "=============================================="
echo "Modeling ITL Perf Benchmark (aiperf)"
echo "=============================================="
echo "Endpoint: ${ENDPOINT}"
echo "Model: ${MODEL_NAME}"
echo "Trace Input: ${TRACE_INPUT}"
echo "K: ${K}"
echo "Concurrencies: ${CONCURRENCIES}"
echo "TTFT Threshold: ${TTFT_THRESHOLD}ms"
echo "ITL Threshold: ${ITL_THRESHOLD}ms"
echo "Tokenizer Path: ${TOKENIZER_PATH}"
echo "ISL Block Size: ${ISL_BLOCK_SIZE}"
echo "Run Cadence Ramp: ${RUN_CADENCE_RAMP}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Extra Args: ${EXTRA_ARGS[*]}"
fi
echo "=============================================="

if [ ! -d "${TRACE_INPUT}" ] && [ ! -f "${TRACE_INPUT}" ]; then
    echo "ERROR: Trace input not found: ${TRACE_INPUT}"
    exit 1
fi

SINGLE_TRACE_FILE=0
if [ -f "${TRACE_INPUT}" ]; then
    SINGLE_TRACE_FILE=1
fi

AIPERF_SPEC="${AIPERF_PACKAGE:-aiperf}"
AIPERF_VENV="/tmp/aiperf-${SLURM_JOB_ID:-$$}"

echo "Setting up aiperf environment: ${AIPERF_SPEC}"
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv "${AIPERF_VENV}"
uv pip install -p "${AIPERF_VENV}" "${AIPERF_SPEC}" tiktoken
export PATH="${AIPERF_VENV}/bin:${PATH}"
echo "aiperf $(aiperf --version 2>/dev/null || echo 'installed') in ${AIPERF_VENV}"

MODEL_BASE_NAME="${MODEL_NAME##*/}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

# Each profile-k{K}-c{C}.jsonl is self-contained: it starts with a warmup header
# of (C + 1) rows (the C background lineages + the injection prefix), a blank
# separator line, then the measured rows. We run one aiperf invocation per file
# with --warmup-request-count so the header primes the engine KV cache (shared
# file => shared trace_id => shared generated content with the measured rows)
# and is excluded from the reported statistics. There is no separate prime pass.

IFS=',' read -r -a CONCURRENCY_LIST <<< "${CONCURRENCIES}"

start_all_profiling

for C in "${CONCURRENCY_LIST[@]}"; do
    if [ "${SINGLE_TRACE_FILE}" -eq 1 ]; then
        SERVER_CONCURRENCY=${C}
        WARMUP_COUNT=${C}
        RUN_FILES=("threshold:${TRACE_INPUT}")
    else
        SERVER_CONCURRENCY=$((C + EXTRA_INJECTION_SLOTS))
        # Warmup header = C background lineages + 1 injection prefix (manifest
        # warmup_request_count_by_file). aiperf consumes the first WARMUP_COUNT
        # rows as its warmup phase and excludes them from reported statistics.
        WARMUP_COUNT=$((C + 1))

        # One self-contained file per run. The cadence ramp is just another such
        # file, appended only when enabled.
        RUN_FILES=("profile:${TRACE_INPUT}/profile-k${K}-c${C}.jsonl")
        if [[ "${RUN_CADENCE_RAMP}" == "1" || "${RUN_CADENCE_RAMP}" == "true" ]]; then
            RAMP_FILE="${TRACE_INPUT}/cadence-ramp-k${K}-c${C}.jsonl"
            if [ -f "${RAMP_FILE}" ]; then
                RUN_FILES+=("cadence_ramp:${RAMP_FILE}")
            else
                echo "No cadence-ramp trace for K=${K}, C=${C}; skipping"
            fi
        fi
    fi

    for ENTRY in "${RUN_FILES[@]}"; do
        KIND="${ENTRY%%:*}"
        INPUT_FILE="${ENTRY#*:}"
        if [ ! -f "${INPUT_FILE}" ]; then
            echo "ERROR: Trace file not found: ${INPUT_FILE}"
            exit 1
        fi

        # Blank separator lines are skipped by the loader; count non-blank rows.
        TOTAL_ROWS=$(grep -cE '[^[:space:]]' "${INPUT_FILE}")
        REQUEST_COUNT=$((TOTAL_ROWS - WARMUP_COUNT))
        if [ "${KIND}" = "threshold" ]; then
            THRESHOLD_NAME=$(basename "${INPUT_FILE}" .jsonl)
            RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_p2_${THRESHOLD_NAME}_${TIMESTAMP}"
        elif [ "${KIND}" = "profile" ]; then
            RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_p0_k${K}_c${C}_${TIMESTAMP}"
        else
            RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_p0_${KIND}_k${K}_c${C}_${TIMESTAMP}"
        fi
        mkdir -p "${RUN_ARTIFACT_DIR}"

        echo ""
        echo "=============================================="
        echo "Running modeling_itl_perf ${KIND}: K=${K}, concurrency=${C}"
        echo "=============================================="
        echo "Input File: ${INPUT_FILE}"
        echo "Warmup Requests: ${WARMUP_COUNT} (primes KV; excluded from stats by aiperf)"
        echo "Measured Requests: ${REQUEST_COUNT}"
        echo "AIPerf Concurrency: ${SERVER_CONCURRENCY}"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting ${KIND}"

        aiperf profile \
            -m "${MODEL_NAME}" \
            --tokenizer "${TOKENIZER_PATH}" \
            --tokenizer-trust-remote-code \
            --input-file "${INPUT_FILE}" \
            --custom-dataset-type mooncake_trace \
            --dataset-sampling-strategy sequential \
            --isl-block-size "${ISL_BLOCK_SIZE}" \
            --url "${ENDPOINT}" \
            --endpoint-type chat \
            --streaming \
            --extra-inputs ignore_eos:true \
            --concurrency "${SERVER_CONCURRENCY}" \
            --warmup-request-count "${WARMUP_COUNT}" \
            --request-count "${REQUEST_COUNT}" \
            --random-seed 42 \
            --ui simple \
            --artifact-dir "${RUN_ARTIFACT_DIR}" \
            "${SERVER_METRICS_ARGS[@]}" \
            "${EXTRA_ARGS[@]}"

        echo "$(date '+%Y-%m-%d %H:%M:%S') - ${KIND} complete: K=${K}, C=${C}"
        ls -la "${RUN_ARTIFACT_DIR}" 2>/dev/null || true
    done
done

stop_all_profiling

echo ""
echo "=============================================="
echo "Modeling ITL Perf Benchmark Complete"
echo "Results saved to: ${ARTIFACT_DIR}"
echo "=============================================="
