# G2/offload follow-up probes

These recipes implement the follow-up plan from
`/localhome/local-karenc/dynamo-workflows/dynamo/disagg/offloading/offload_g2_probe_recap.md`.

The required first pass is intentionally agg-only:

| recipe | priority | matched baseline | change | what to look for |
|---|---|---|---|---|
| `01_agg_offload_nogate_400g_12gpu_tep4x3.yaml` | P1 | `2209688`, plus no-offload agg `2130303` | Same eager agg offload probe, but `cpu_bytes_to_use=429496729600` and c96/c80 sweep | If CPU->GPU load remains near zero, 120GiB G2 capacity was not the blocker. If CPU->GPU rises and latency improves, the old probe was capacity-limited. If CPU->GPU rises but perf regresses, offload lookup/write/pin overhead dominates. |
| `02_agg_offload_nogate_120g_logs_12gpu_tep4x3.yaml` | P2 | `2209688` | Exact 120GiB eager agg rerun, named for log/metric capture | Use this as the write-through cost rerun. Compare G1 pressure, offload load/store bytes, offload load/store time, and any pin-duration metric if the container includes it. |

Optional follow-ups are included but not launched by `run.sh`:

| recipe | priority | matched baseline | change | why run it |
|---|---|---|---|---|
| `03_agg_offload_threshold2_400g_12gpu_tep4x3.yaml` | P3 | `2130303` / `2209688` | 400GiB G2 plus `store_threshold=2`, `max_tracker_size=262144` | Tests whether eager write-through is the problem rather than G2 itself. |
| `04_disagg_tuned_offload_400g_14gpu_tep4x2p_tep2x3d.yaml` | P4 | tuned disagg offload `2180791` / `2180792` | Same tuned NIXL-first decode offload, `block_size=256`, but 400GiB G2 | Lower priority capacity check for the disagg tuned-offload path. |

Notes:

- `cpu_bytes_to_use=429496729600` is nominal 400GiB per vLLM engine group, not 400GiB per TP rank.
- `store_threshold=0` intentionally disables store admission filtering; this is the eager write-through stress case.
- `offload_prompt_only=true` stays consistent with the earlier agg/disagg probes.
- `kv-cache-metrics=true` is enabled in all recipes.

Launch the required first-pass jobs:

```bash
bash /localhome/local-karenc/srt-slurm/recipes/v1_isl_and_load/cc-traces-weka-with-subagents-052726-256k/offload_g2_followup/run.sh
```

