#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ENDPOINT="${1:?usage: bench.sh ENDPOINT MODEL}"
MODEL="${2:?usage: bench.sh ENDPOINT MODEL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/profiling.sh
source "${SCRIPT_DIR}/../lib/profiling.sh"
profiling_init_from_env

PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-/logs/profiles}"
MARKER="${PROFILE_OUTPUT_DIR}/profiling-smoke-passed.txt"
REQUEST_OUTPUT="/tmp/profiling-smoke-response.json"
mkdir -p "${PROFILE_OUTPUT_DIR}"

cleanup() {
    stop_all_profiling
}
trap cleanup EXIT

request() {
    local phase="$1"
    local max_tokens="$2"
    local http_code
    local payload

    payload="$(printf \
        '{"model":"%s","messages":[{"role":"user","content":"Reply briefly."}],"temperature":0,"max_tokens":%s,"ignore_eos":true}' \
        "${MODEL}" "${max_tokens}")"
    http_code="$(curl --silent --show-error \
        --max-time 180 \
        --output "${REQUEST_OUTPUT}" \
        --write-out '%{http_code}' \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        "${ENDPOINT}/v1/chat/completions")"

    if [[ "${http_code}" != "200" ]]; then
        echo "ERROR: ${phase} request returned HTTP ${http_code}" >&2
        cat "${REQUEST_OUTPUT}" >&2 || true
        return 1
    fi
    echo "PASS: ${phase} request returned HTTP 200"
}

wait_for_worker_log() {
    local pattern="$1"
    local timeout_seconds="${2:-180}"
    local deadline=$((SECONDS + timeout_seconds))

    while ((SECONDS < deadline)); do
        if grep -R -F -q \
            --include='*_agg_w*.out*' \
            -- "${pattern}" /logs 2>/dev/null; then
            echo "PASS: observed worker log: ${pattern}"
            return 0
        fi
        sleep 2
    done

    echo "ERROR: timed out waiting for worker log: ${pattern}" >&2
    find /logs -maxdepth 2 -type f -name '*_agg_w*.out*' -print >&2 || true
    return 1
}

echo "Profiling smoke: proving serving before, during, and after auto-stop"
echo "Profile output mount: ${PROFILE_OUTPUT_DIR}"

request "pre-profile-1" 2
request "pre-profile-2" 2

start_all_profiling

# ignore_eos keeps this request alive long enough to cross the small
# iteration-bounded capture window in the smoke recipe.
request "capture-driving" 32
wait_for_worker_log "Starting profiler after delay..."
wait_for_worker_log "Max profiling iterations reached. Stopping profiler..."
wait_for_worker_log "Profiler stopped successfully."

request "post-auto-stop-1" 4
request "post-auto-stop-2" 4
request "post-auto-stop-3" 4

# This clears vLLM's still-active bookkeeping after its automatic CUDA stop.
stop_all_profiling
request "post-explicit-cleanup" 4

{
    echo "status=passed"
    echo "completed_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "profile_output_dir=${PROFILE_OUTPUT_DIR}"
    echo "expected_report_glob=${PROFILE_OUTPUT_DIR}/agg/*.nsys-rep"
    echo "note=The report may finalize only after srt-slurm cleans up the Nsys-wrapped worker."
} > "${MARKER}"

trap - EXIT
echo "PASS: profiling lifecycle smoke completed"
echo "Marker: ${MARKER}"
