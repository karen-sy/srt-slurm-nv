#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Modeling perf controlled trace benchmark using aiperf.
#
# Usage:
#   bench.sh ENDPOINT MODEL_NAME TRACE_DIR K CONCURRENCIES \
#     [TTFT_THRESHOLD] [ITL_THRESHOLD] [TOKENIZER_PATH] [EXTRA_AIPERF_ARGS...]
#
# TRACE_DIR must contain prime.jsonl plus profile-k${K}-c${C}.jsonl files.
# The trace rows intentionally have no timestamp/delay fields; this script runs
# closed-loop by concurrency and uses request-count to consume each file once.

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
TRACE_DIR=$3
K=$4
CONCURRENCIES=${5:-"16,32,64"}
TTFT_THRESHOLD=${6:-2000}
ITL_THRESHOLD=${7:-25}
TOKENIZER_PATH=${8:-"/model"}
shift 8 2>/dev/null || true
EXTRA_ARGS=("$@")

ISL_BLOCK_SIZE="${AIPERF_ISL_BLOCK_SIZE:-64}"
PRIME_CONCURRENCY="${MODELING_ITL_PERF_PRIME_CONCURRENCY:-1}"
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
echo "Trace Dir: ${TRACE_DIR}"
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

if [ ! -d "${TRACE_DIR}" ]; then
    echo "ERROR: Trace directory not found: ${TRACE_DIR}"
    exit 1
fi

PRIME_FILE="${TRACE_DIR}/prime.jsonl"
if [ ! -f "${PRIME_FILE}" ]; then
    echo "ERROR: Prime trace not found: ${PRIME_FILE}"
    exit 1
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

PRIME_COUNT=$(wc -l < "${PRIME_FILE}" | tr -d ' ')
PRIME_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_p0_prime_k${K}_${TIMESTAMP}"
mkdir -p "${PRIME_DIR}"

echo ""
echo "=============================================="
echo "Priming cache"
echo "=============================================="
echo "Prime File: ${PRIME_FILE}"
echo "Prime Requests: ${PRIME_COUNT}"
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting prime run"

aiperf profile \
    -m "${MODEL_NAME}" \
    --tokenizer "${TOKENIZER_PATH}" \
    --tokenizer-trust-remote-code \
    --input-file "${PRIME_FILE}" \
    --custom-dataset-type mooncake_trace \
    --dataset-sampling-strategy sequential \
    --isl-block-size "${ISL_BLOCK_SIZE}" \
    --url "${ENDPOINT}" \
    --endpoint-type chat \
    --streaming \
    --extra-inputs ignore_eos:true \
    --concurrency "${PRIME_CONCURRENCY}" \
    --request-count "${PRIME_COUNT}" \
    --random-seed 42 \
    --ui simple \
    --artifact-dir "${PRIME_DIR}"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Prime complete"

IFS=',' read -r -a CONCURRENCY_LIST <<< "${CONCURRENCIES}"

start_all_profiling

for C in "${CONCURRENCY_LIST[@]}"; do
    PROFILE_FILE="${TRACE_DIR}/profile-k${K}-c${C}.jsonl"
    if [ ! -f "${PROFILE_FILE}" ]; then
        echo "ERROR: Profile trace not found: ${PROFILE_FILE}"
        exit 1
    fi

    REQUEST_COUNT=$(wc -l < "${PROFILE_FILE}" | tr -d ' ')
    SERVER_CONCURRENCY=$((C + EXTRA_INJECTION_SLOTS))
    RUN_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_p0_k${K}_c${C}_${TIMESTAMP}"
    mkdir -p "${RUN_ARTIFACT_DIR}"

    echo ""
    echo "=============================================="
    echo "Running modeling_itl_perf profile: K=${K}, concurrency=${C}"
    echo "=============================================="
    echo "Profile File: ${PROFILE_FILE}"
    echo "Profile Requests: ${REQUEST_COUNT}"
    echo "AIPerf Concurrency: ${SERVER_CONCURRENCY} (${C} background + ${EXTRA_INJECTION_SLOTS} injection slot)"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting profile"

    aiperf profile \
        -m "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_PATH}" \
        --tokenizer-trust-remote-code \
        --input-file "${PROFILE_FILE}" \
        --custom-dataset-type mooncake_trace \
        --dataset-sampling-strategy sequential \
        --isl-block-size "${ISL_BLOCK_SIZE}" \
        --url "${ENDPOINT}" \
        --endpoint-type chat \
        --streaming \
        --extra-inputs ignore_eos:true \
        --concurrency "${SERVER_CONCURRENCY}" \
        --request-count "${REQUEST_COUNT}" \
        --random-seed 42 \
        --ui simple \
        --artifact-dir "${RUN_ARTIFACT_DIR}" \
        "${SERVER_METRICS_ARGS[@]}" \
        --goodput "time_to_first_token:${TTFT_THRESHOLD} inter_token_latency:${ITL_THRESHOLD}" \
        "${EXTRA_ARGS[@]}"

    echo "$(date '+%Y-%m-%d %H:%M:%S') - Profile complete: K=${K}, C=${C}"
    ls -la "${RUN_ARTIFACT_DIR}" 2>/dev/null || true

    if [[ "${RUN_CADENCE_RAMP}" == "1" || "${RUN_CADENCE_RAMP}" == "true" ]]; then
        RAMP_FILE="${TRACE_DIR}/cadence-ramp-k${K}-c${C}.jsonl"
        if [ ! -f "${RAMP_FILE}" ]; then
            echo "No cadence-ramp trace for K=${K}, C=${C}; skipping"
            continue
        fi

        RAMP_REQUEST_COUNT=$(wc -l < "${RAMP_FILE}" | tr -d ' ')
        RAMP_ARTIFACT_DIR="${ARTIFACT_DIR}/${MODEL_BASE_NAME}_p0_cadence_ramp_k${K}_c${C}_${TIMESTAMP}"
        mkdir -p "${RAMP_ARTIFACT_DIR}"

        echo ""
        echo "=============================================="
        echo "Running modeling_itl_perf cadence ramp: K=${K}, concurrency=${C}"
        echo "=============================================="
        echo "Ramp File: ${RAMP_FILE}"
        echo "Ramp Requests: ${RAMP_REQUEST_COUNT}"
        echo "AIPerf Concurrency: ${SERVER_CONCURRENCY} (${C} background + ${EXTRA_INJECTION_SLOTS} injection slot)"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting cadence ramp"

        aiperf profile \
            -m "${MODEL_NAME}" \
            --tokenizer "${TOKENIZER_PATH}" \
            --tokenizer-trust-remote-code \
            --input-file "${RAMP_FILE}" \
            --custom-dataset-type mooncake_trace \
            --dataset-sampling-strategy sequential \
            --isl-block-size "${ISL_BLOCK_SIZE}" \
            --url "${ENDPOINT}" \
            --endpoint-type chat \
            --streaming \
            --extra-inputs ignore_eos:true \
            --concurrency "${SERVER_CONCURRENCY}" \
            --request-count "${RAMP_REQUEST_COUNT}" \
            --random-seed 42 \
            --ui simple \
            --artifact-dir "${RAMP_ARTIFACT_DIR}" \
            "${SERVER_METRICS_ARGS[@]}" \
            --goodput "time_to_first_token:${TTFT_THRESHOLD} inter_token_latency:${ITL_THRESHOLD}" \
            "${EXTRA_ARGS[@]}"

        echo "$(date '+%Y-%m-%d %H:%M:%S') - Cadence ramp complete: K=${K}, C=${C}"
        ls -la "${RAMP_ARTIFACT_DIR}" 2>/dev/null || true
    fi
done

stop_all_profiling

echo ""
echo "=============================================="
echo "Modeling ITL Perf Benchmark Complete"
echo "Results saved to: ${ARTIFACT_DIR}"
echo "=============================================="
