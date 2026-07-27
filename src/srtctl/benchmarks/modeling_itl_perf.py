# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling performance controlled trace benchmark runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, AIPerfBenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("modeling_itl_perf")
class ModelingItlPerfRunner(AIPerfBenchmarkRunner):
    """Run controlled traces for modeling performance experiments.

    This benchmark accepts either:

    - a P0 trace directory with ``profile-k{K}-c{C}.jsonl`` files; or
    - one self-contained P2 threshold-projection JSONL.

    Existing BenchmarkConfig fields are reused to avoid P0-specific schema
    churn: benchmark.trace_file is the directory or file, benchmark.isl is K,
    and benchmark.concurrencies selects the concurrency.
    """

    @property
    def name(self) -> str:
        return "Modeling-ITL-Perf"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/modeling_itl_perf/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "modeling_itl_perf")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        b = config.benchmark

        if not b.trace_file:
            errors.append(
                "benchmark.trace_file is required for modeling_itl_perf "
                "and must point to the generated trace directory"
            )

        if b.isl is None:
            errors.append(
                "benchmark.isl is required for modeling_itl_perf and is "
                "used as the prefill chunk size K"
            )

        if b.concurrencies is None:
            errors.append("benchmark.concurrencies is required for modeling_itl_perf")

        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        b = config.benchmark
        endpoint = f"http://localhost:{runtime.frontend_port}"
        model_name = config.served_model_name or config.model.path

        concurrencies = b.concurrencies
        if isinstance(concurrencies, list):
            concurrencies = ",".join(str(c) for c in concurrencies)

        ttft_threshold = b.ttft_threshold_ms or 2000
        itl_threshold = b.itl_threshold_ms or 25
        tokenizer_path = str(runtime.model_path) if runtime.is_hf_model else "/model"

        cmd = [
            "bash",
            self.script_path,
            endpoint,
            model_name,
            b.trace_file or "",
            str(b.isl or ""),
            str(concurrencies or ""),
            str(ttft_threshold),
            str(itl_threshold),
            tokenizer_path,
        ]

        self.append_aiperf_args(cmd, config)

        return cmd

    def get_environment(self, config: SrtConfig, runtime: RuntimeContext) -> dict[str, str]:
        del runtime
        return dict(config.benchmark.env)
