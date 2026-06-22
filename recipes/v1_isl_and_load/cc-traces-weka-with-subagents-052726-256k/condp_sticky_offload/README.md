# Conditional-disagg sticky + decode-only offload experiments

Reference table: `/localhome/local-karenc/dynamo-workflows/dynamo/disagg/conditional_prefill/CONDP_MAIN_PERF_TABLE_0605.md`.

Goal: push the 14GPU `tep4x2p_tep2x3d` sticky/conditional-disagg config at c96 and c80 by adding CPU KV headroom only on decode workers, without repeating the old all-worker offload failure mode.

Baseline anchor:

- `2178348`: no-offload sticky qNone, decode gate 0.90, c80.
- `2178352`: no-offload sticky qNone, decode gate 0.90, c96.
- Current symptom: best tok/s/GPU in the table, better TTFT than old no-offload, but decode total hit remains about 6%, so decode KV residency is still poor.

Why these recipes differ from old offload runs:

- Old offload used `MultiConnector(NixlConnector + OffloadingConnector)` on both prefill and decode.
- Old runs moved tens of TB GPU-to-CPU, mostly stores/demotions, with little CPU-to-GPU recall.
- These recipes keep prefill as plain NIXL producer and enable CPU offload only on decode.
- Decode offload uses `store_threshold: 2` so one-off prompt blocks are not immediately demoted to CPU.
- `offload_prompt_only: true` is kept to avoid storing generated decode-token KV.

Recipes:

1. `01_decode_offload_lru_14gpu_tep4x2p_tep2x3d.yaml`
   - First test.
   - Decode-only CPU offload with `store_threshold: 2`, `max_tracker_size: 262144`, `eviction_policy: lru`.

2. `02_decode_offload_arc_14gpu_tep4x2p_tep2x3d.yaml`
   - Same as #1, but `eviction_policy: arc`.
   - Tests retention policy after admission filtering.

3. `03_decode_offload_lru_block256_14gpu_tep4x2p_tep2x3d.yaml`
   - Same as #1, but `block_size: 256` for the OffloadingConnector.
   - Tests coarser CPU-offload chunks; this is higher risk because CPU hits become chunkier.

Shared routing/config:

- `router-conditional-disagg: true`
- `router-conditional-disagg-policy: isl_or_load`
- `router-conditional-disagg-prefill-busy-threshold: 16`
- `router-conditional-disagg-decode-busy-threshold: 0.90`
- `router-queue-threshold: None`
- `concurrency: [96, 80]`

Metrics to check:

- `tok/s/GPU`, TTFT p50/avg, ITL p50/avg.
- `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries`.
- `vllm:external_prefix_cache_hits`, `vllm:external_prefix_cache_queries`.
- `vllm:kv_offload_store_bytes`, `vllm:kv_offload_store_time`, `vllm:kv_offload_store_size`.
- `vllm:kv_offload_load_bytes`, `vllm:kv_offload_load_time`, `vllm:kv_offload_load_size`.
- Legacy equivalents: `vllm:kv_offload_total_bytes{transfer_type="GPU_to_CPU"|"CPU_to_GPU"}`.
- `vllm:kv_offload_cpu_cache_usage_perc`.
- `vllm:kv_offload_stores_skipped`.
- `vllm:kv_cache_stored_blocks` and `vllm:kv_cache_duplicate_stored_blocks` if the patched vLLM image and `kv-cache-metrics` are active.

Success criteria:

- GPU-to-CPU store traffic is far lower than old all-worker offload runs.
- CPU-to-GPU load traffic is nonzero and corresponds to useful reuse.
- Decode total hit improves over the no-offload c96/c80 anchors.
- TTFT does not regress toward the old gated-offload `~36s` average.
- ITL improves materially from the no-offload `~32ms` average.
- tok/s/GPU stays close to, or beats, `2178352`.

Launch:

```bash
bash /localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/condp_sticky_offload/run.sh
```
