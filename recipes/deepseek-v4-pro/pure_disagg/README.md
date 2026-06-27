# DeepSeek-V4-Pro vLLM Pure-Disagg Trace-Replay-SA Recipes

These recipes are first-pass vLLM pure-disagg baselines for
`deepseek-ai/DeepSeek-V4-Pro` on the SemiAnalysis trace-replay-SA workload.
They intentionally do not enable conditional-disagg, sticky decode affinity,
offload, or bidirectional NIXL. The goal is to locate vLLM pure disagg relative
to known TRT-LLM and SGLang topology points before adding policy features.

The worker flags keep `--tokenizer-mode deepseek_v4`; Dynamo-side parser
settings are supplied through `DYN_REASONING_PARSER=deepseek_v4` and
`DYN_TOOL_CALL_PARSER=deepseek_v4`, matching the Qwen trace-replay recipes.
The standalone vLLM OpenAI parser flags are not used here.

## Recipes

| file | topology | GPUs | concurrency | reference point | what to look for |
|---|---:|---:|---:|---|---|
| `01_vllm_dsv4_ctx2dep8_gen1dep16_c224_c384.yaml` | `ctx2dep8-gen1dep16` | 32 | 224, 384 | TRT 32GPU anchor sweep | Covers the lowest and highest TRT concurrency points for this topology. |
| `02_vllm_dsv4_ctx3dep8_gen1dep16_c576.yaml` | `ctx3dep8-gen1dep16` | 40 | 576 | TRT 40GPU decode-room anchor | Tests whether extra prefill workers lift throughput without decode becoming the limiter. |
| `03_vllm_dsv4_ctx6dep4_gen1dep16_c768.yaml` | `ctx6dep4-gen1dep16` | 40 | 768 | SGLang-style DEP4 prefill granularity | Same 24P/16D GPU split as recipe 02, but six DEP4 prefill workers instead of three DEP8 workers. |
| `04_vllm_dsv4_ctx3dep4_gen1dep8_c512.yaml` | `ctx3dep4-gen1dep8` | 20 | 512 | Cheaper SGLang-shape sanity point | Lower-cost high-concurrency smoke/perf point; useful if the 40GPU runs are slow to iterate. |

## Notes

- `model.path` uses the `DeepSeek-V4-Pro` alias; resolve it in `srtslurm.yaml`
  for the target cluster.
- `served-model-name: deepseek-ai/DeepSeek-V4-Pro` is set in both worker configs
  so the trace-replay-SA client uses the same model name the workers serve.
- `data-parallel-size-local`, `data-parallel-start-rank`, and
  `data-parallel-address` are intentionally omitted. srt-slurm injects
  `--data-parallel-rank`, `--data-parallel-address`, and
  `--data-parallel-rpc-port` for each DP worker process.
- These recipes are shaped for the same long-context SA traces as the Qwen
  conditional-disagg work: `max-model-len: 262144`, explicit prefix caching,
  prefill `max-num-batched-tokens: 32768`, and decode
  `max-num-batched-tokens: 4352`.
- Kubernetes-specific launch environment from Dynamo sample recipes is
  intentionally omitted; srt-slurm should provide cluster transport/runtime
  environment separately.
