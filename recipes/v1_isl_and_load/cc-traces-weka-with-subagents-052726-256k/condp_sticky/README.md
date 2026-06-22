# Conditional-disagg sticky/decode-affinity experiments

Reference table: `/localhome/local-karenc/dynamo-workflows/dynamo/disagg/conditional_prefill/CONDP_MAIN_PERF_TABLE_0605.md`.

These recipes are intended to answer three questions from the 0605 perf table:

1. Can conditional-disagg decode-overlap affinity recover decode-local reuse on the 14-GPU champion topology?
   - Anchor: `2108697`, no-offload `tep4x2p_tep2x3d`, c96, `isl_or_load`, qNone/pbusy16.
   - Baseline symptoms: good tok/s/GPU, high prefill hit, but decode local only ~11.8% and combined local ~5%.

2. Does a decode-side circuit breaker prevent the cache-full/decode-throttle regime without throwing away the TTFT win?
   - Anchor: `2108695` vs `2108697`, same topology/c96 but q16 vs qNone.
   - New variable: `--router-conditional-disagg-decode-busy-threshold` at 0.90 and 0.95.

3. Is the 20-GPU high-decode-local case mostly a capacity/headroom effect, or can affinity still move it?
   - Anchor: `2059303`, no-offload `tep4x2p_tep4x3d`, c96, high decode-local reuse (~75.5%).
   - Expectation: smaller gain than 14-GPU; useful no-regression check.

Here "sticky" means the decode cache-affinity behavior implemented in the
current Dynamo branch: when conditional-disagg is enabled and
the default router overlap credit is inherited, the decode router can score
candidate decode workers by KV overlap instead of staying purely load-only.

Important run-readiness notes:

- These recipes use the renamed current flags: `router-conditional-disagg*` instead of the stale `router-conditional-prefill*` names.
- They intentionally rely on inherited base router overlap credit: conditional-disagg leaves the per-request decode override unset, so it inherits the base router default.
- The `model.container` value is intentionally inherited from the latest condp vLLM recipes. Replace it with the image built from the current Dynamo branch if the old image does not include the renamed flags.

Suggested launch order:

1. `01_nooffload_14gpu_tep4x2p_tep2x3d_affinity_isolate_c96.yaml`
   - Single matched c96 comparison against `2108697`.
   - Best first sanity check for decode-overlap affinity.

2. `02_nooffload_14gpu_tep4x2p_tep2x3d_decode_gate_c96.yaml`
   - 4-run grid: qNone/q16 x decode gate 0.90/0.95.
   - Tests whether headroom protection helps the c96 decode-local collapse.

3. `03_nooffload_20gpu_tep4x2p_tep4x3d_affinity_contrast_c96.yaml`
   - Single c96 contrast against `2059303`.
   - Should not regress the high decode-local topology.

4. `04_nooffload_14gpu_tep4x2p_tep2x3d_concurrency_shape_decode_gate.yaml`
   - c60/c80/c96 with qNone, decode gate 0.95.
   - Checks whether the policy shifts the previous “c96 too much churn” shape.

Primary metrics to compare:

- tok/s/GPU
- TTFT p50/avg
- ITL p50/avg
- prefill hit
- decode local / external hit
- combined local hit
- decode KV availability / decode KV usage over time
- router logs for conditional-disagg decision, overlap tokens/blocks, decode gate decision, and selected decode worker
