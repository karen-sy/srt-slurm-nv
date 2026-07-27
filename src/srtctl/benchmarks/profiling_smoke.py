# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end profiling lifecycle smoke benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

from srtctl.benchmarks.base import SCRIPTS_DIR, BenchmarkRunner, register_benchmark

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext
    from srtctl.core.schema import SrtConfig


@register_benchmark("profiling_smoke")
class ProfilingSmokeRunner(BenchmarkRunner):
    """Verify that an iteration-bounded profile does not stop serving."""

    @property
    def name(self) -> str:
        return "Profiling-Smoke"

    @property
    def script_path(self) -> str:
        return "/srtctl-benchmarks/profiling_smoke/bench.sh"

    @property
    def local_script_dir(self) -> str:
        return str(SCRIPTS_DIR / "profiling_smoke")

    def validate_config(self, config: SrtConfig) -> list[str]:
        errors = []
        if not config.profiling.is_nsys:
            errors.append("profiling_smoke requires profiling.type: nsys")
        if config.resources.num_agg == 0:
            errors.append("profiling_smoke currently requires aggregated workers")
        if config.profiling.aggregated is None:
            errors.append("profiling_smoke requires profiling.aggregated")
        return errors

    def build_command(
        self,
        config: SrtConfig,
        runtime: RuntimeContext,
    ) -> list[str]:
        endpoint = f"http://localhost:{runtime.frontend_port}"
        model_name = config.served_model_name or config.model.path
        return ["bash", self.script_path, endpoint, model_name]
