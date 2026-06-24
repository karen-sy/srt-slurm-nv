# Conditional-disagg sticky + vLLM NIXL bidirectional D->P pull

Reference table: `/localhome/local-karenc/dynamo-workflows/dynamo/disagg/conditional_prefill/CONDP_MAIN_PERF_TABLE_0605.md`.

Goal: test the preliminary vLLM NIXL bidirectional KV path, where Dynamo caches decode-side `kv_transfer_params` by session and injects them into later prefill requests so P can pull existing KV from D before computing only the new tokens.

Shared deltas from the prior sticky recipes:

- Prefill NIXL uses `kv_role=kv_producer` with `kv_connector_extra_config.bidirectional_kv_xfer=true`.
- Decode NIXL uses `kv_role=kv_consumer` with `kv_connector_extra_config.bidirectional_kv_xfer=true`.
- Frontend env enables Dynamo's session KV cache with `DYN_ENABLE_VLLM_NIXL_BIDIRECTIONAL_KV=1`.
- `DYN_VLLM_NIXL_BIDIRECTIONAL_KV_TTL_SECS=450`, slightly below vLLM's default `decoder_kv_blocks_ttl=480`.
- All recipes keep `kv-cache-metrics: true` and c96 before c80 in the sweep.

Important caveat: this branch keys the cache from `x-dynamo-session-id` / session-affinity context. If the benchmark client does not emit that header for turns in the same conversation, these recipes will enable the connector but the D->P path will be inert. Confirm in `frontend.out` by looking for `cached decode KV transfer params` and `injected cached decode KV transfer params` log lines.

Recipes:

| recipe | matched baseline(s) | why it stays | what to look for / possible outcomes |
|---|---|---|---|
| `01_nooffload_bidi_14gpu_tep4x2p_tep2x3d_decode_gate.yaml` | `2178352` c96: 16.61k tok/s/GPU; `2178348` c80: 14.80k | Current 14GPU champion shape. | Expect modest TTFT gain at best because prefill local hit was already high and c96 decode-local residency was poor. If TTFT improves without ITL/tok regression, D->P is still useful in the champion config; large tok/s gain is unlikely. |
| `02_nooffload_bidi_16gpu_tep4x2p_tep4x2d_decode_gate.yaml` | `2182118` c96: 16.25k; `2182119` c80: 14.07k | Cleanest D->P target: lower prefill-local hit, high decode-local residency, and decode KV has headroom. | Success is lower TTFT p50/avg plus lower prefill compute/queue. Tok/s/GPU is upside, not the base expectation, unless residual prefill pressure was limiting throughput. |
| `03_nooffload_bidi_12gpu_tep4x1p_tep4x2d_prefill_scarce.yaml` | No exact prior match. Closest high-residency anchor: `2182118` c96: 16.25k; `2182119` c80: 14.07k | Best test for the desired regime: drop one TP4 prefill worker while keeping the high-residency `2xTP4` decode pool. | Good outcome: total throughput remains near the 16GPU anchor, so per-GPU throughput rises, and TTFT does not explode. Bad outcome: one prefill worker cannot handle new sessions/cache misses even with D->P. |
| `05_offload_bidi_lru_14gpu_tep4x2p_tep2x3d.yaml` | `2180786` c96: 16.42k; `2180787` c80: 14.87k | Matched tuned offload isolate. | Good outcome: TTFT improves over `2180786` and ITL holds. Bad outcome: offload overhead still dominates and remains below no-offload sticky `2178352`. Also check CPU->GPU load/store ratio and stale remote-block failures. |

Deferred:

| recipe | why deferred |
|---|---|
| `deferred_nooffload_bidi_16gpu_tep4x1p_tep4x3d_prefill_scarce.yaml` | Interesting high-decode-headroom allocation, but less essential than the cleaner `1xTP4 P + 2xTP4 D` test because it changes decode capacity too. |

Primary metrics to compare:

- TTFT p50/avg first; PR data suggests this is the main expected win.
- tok/s/GPU second; expect smaller movement unless prefill queueing collapses.
- Prefill prompt source split: local vs external D->P pull vs compute, if exposed.
- Decode prompt source split: local / external / compute.
- `vllm:nixl_remote_blocks_before_prefix_trim` and `after_prefix_trim`.
- Any new bidirectional/cache log lines in `frontend.out` and vLLM logs.
- Error/fallback rate for stale remote block IDs.

Launch:

```bash
bash /localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/condp_sticky_bidi/run.sh
```
