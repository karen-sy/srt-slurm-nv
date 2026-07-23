#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter sweep generation for YAML configs.

This module generates multiple job configs from a sweep configuration by
expanding all combinations of sweep parameters.
"""

import copy
import itertools
from typing import Any


def expand_template(template: Any, values: dict[str, Any]) -> Any:
    """Recursively expand template strings with values.

    Args:
        template: Template object (dict, list, str, or other)
        values: Dictionary of parameter values to substitute

    Returns:
        Expanded template with {param} placeholders replaced
    """
    if isinstance(template, dict):
        return {k: expand_template(v, values) for k, v in template.items()}
    elif isinstance(template, list):
        return [expand_template(item, values) for item in template]
    elif isinstance(template, str):
        result = template
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            # Handle list values specially - convert to comma-separated string or keep as list
            if isinstance(value, list):
                # For YAML lists, we want to keep them as lists, not convert to string
                if placeholder in result and result == placeholder:
                    # If the entire string is just the placeholder, replace with the list
                    return value
                else:
                    # If it's embedded in a string, convert to comma-separated
                    result = result.replace(placeholder, ",".join(str(v) for v in value))
            else:
                result = result.replace(placeholder, str(value))
        return result
    else:
        return template


def generate_sweep_configs(sweep_config: dict) -> list[tuple[dict, dict]]:
    """Generate all job configs from a sweep configuration.

    Args:
        sweep_config: Config dict with 'sweep' section defining parameters

    Returns:
        List of (expanded_config, param_values) tuples
    """
    if "sweep" not in sweep_config:
        raise ValueError("Sweep config must have 'sweep' section")

    # Apply cluster defaults before sweep expansion
    from srtctl.core.config import load_cluster_config, resolve_config_with_defaults

    cluster_config = load_cluster_config()
    sweep_config = resolve_config_with_defaults(sweep_config, cluster_config)

    # Extract sweep parameters.  The original shorthand form is a cartesian
    # grid for backwards compatibility:
    #
    #   sweep:
    #     a: [1, 2]
    #     b: [3, 4]
    #
    # The schema also accepts an explicit form, which can request zip mode:
    #
    #   sweep:
    #     mode: zip
    #     parameters:
    #       a: [1, 2]
    #       b: [3, 4]
    raw_sweep_params = sweep_config["sweep"]
    if not isinstance(raw_sweep_params, dict):
        raise ValueError("sweep section must be a mapping")

    mode = raw_sweep_params.get("mode", "grid")
    if mode not in ("grid", "zip"):
        raise ValueError(f"sweep.mode must be 'grid' or 'zip', got {mode!r}")

    if "parameters" in raw_sweep_params:
        sweep_params = raw_sweep_params["parameters"]
    else:
        sweep_params = {k: v for k, v in raw_sweep_params.items() if k != "mode"}

    # Generate all combinations
    param_names = list(sweep_params.keys())
    param_values_list = [sweep_params[name] for name in param_names]
    for name, values in zip(param_names, param_values_list, strict=False):
        if not isinstance(values, list):
            raise ValueError(f"Sweep parameter '{name}' must be a list")

    configs = []
    if mode == "zip":
        value_iter = zip(*param_values_list, strict=False)
    else:
        value_iter = itertools.product(*param_values_list)

    for values in value_iter:
        # Create parameter dict for this combination
        params = dict(zip(param_names, values, strict=False))

        # Create a copy of the config without the sweep section
        config = copy.deepcopy(sweep_config)
        del config["sweep"]

        # Expand all template placeholders
        config = expand_template(config, params)

        # Generate a unique name for this config
        param_str = "_".join(f"{k}{v}" for k, v in params.items())
        config["name"] = f"{sweep_config['name']}_{param_str}"

        # Validate and serialize back to dict
        from srtctl.core.schema import SrtConfig

        schema = SrtConfig.Schema()
        validated = schema.load(config)
        config = schema.dump(validated)

        configs.append((config, params))

    return configs
