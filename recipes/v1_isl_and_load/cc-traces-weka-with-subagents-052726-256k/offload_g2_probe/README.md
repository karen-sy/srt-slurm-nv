# G2/offload connector probes

These are the two minimal experiments requested in the offload discussion. They are intentionally c96-only so the launch maps to two jobs.

| recipe | matched baseline | change | what to look for |
|---|---|---|---|
| `01_agg_single_offload_nogate_12gpu_tep4x3.yaml` | agg `tep4x3` no-offload c96, `2130303` | aggregated serving, single `OffloadingConnector`, `store_threshold: 0`, `offload_prompt_only: true` | This tests whether G2 is useful when no NIXL connector can win first, while keeping prompt-only offload consistent with the disagg probe. Expect lots of GPU->CPU prompt-block writes. If G2 is useful, CPU->GPU load bytes and offload hit rate should be nontrivial; if not, Harry's short-reuse/G1-sufficient hypothesis looks stronger. |
| `02_disagg_offload_first_nogate_14gpu_tep4x2p_tep2x3d.yaml` | tuned sticky decode-offload c96, `2180786`; old no-gate offload c96, `2060377` | same conditional-disagg topology as the tuned offload run, but `MultiConnector` order is `OffloadingConnector` then `NixlConnector`, and `store_threshold: 0` | This tests whether NIXL-first ordering hid G2 hits. If order mattered, CPU->GPU load bytes should rise and NIXL external bytes should fall. If throughput drops while G2 hits rise, G2 is slower than NIXL on the critical path. If G2 still sees little load, the missing hits are not just connector order. |

Metrics to check:

- Top-level perf: `tok/s/GPU`, TTFT p50/avg, ITL p50/avg.
- Offload traffic: `vllm:kv_offload_total_bytes{transfer_type="GPU_to_CPU"|"CPU_to_GPU"}` and the newer store/load byte/time metrics if present.
- Offload selectivity: CPU->GPU / GPU->CPU byte ratio.
- Connector source mix: NIXL external bytes/hits versus offload load bytes/hits.
- KV fullness and prefix-cache metrics: `vllm:prefix_cache_hits`, `vllm:prefix_cache_queries`, `vllm:external_prefix_cache_hits`, `vllm:external_prefix_cache_queries`, and `vllm:kv_cache_*` metrics if active.

Launch:

```bash
bash /localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/offload_g2_probe/run.sh
```
