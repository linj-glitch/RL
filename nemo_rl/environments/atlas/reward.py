# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness-gated reward for one CudaGym kernel evaluation.

Shared by the single-turn and agentic paths (the Gym cudagym resources server
vendors the same logic — keep the two in step). A kernel that is not
numerically correct on EVERY workload earns exactly 0; a correct one earns the
correctness weight plus the performance weight scaled by the anchored SOL
score (see ``get_reward``). Format/compile/execute progress is observable in
the metrics but never rewarded.
"""

import math

from .cuda_kernel_utils import KernelEvalResult


def get_reward(
    result: KernelEvalResult,
    weights: dict[str, float],
    perf_reward_config: dict[str, float],
) -> float:
    """Compute the staged scalar reward for a single ``KernelEvalResult``.

    Args:
        result: the populated evaluation outcome (flags + speedup).
        weights: per-stage weights, e.g.
            ``{"format":1,"compiled":2,"executed":2,"correctness":4,"performance":8}``.
        perf_reward_config: ``{"clip_max","clip_min","speedup_ratio"}`` for the
            performance term's log-scale normalization.

    Returns:
        The summed reward. Early-returns at the first failed stage, so the value
        is monotonic in how far the kernel got.
    """
    reward = 0.0

    # Format: the completion parsed into a well-formed Solution.
    if result.formatted:
        reward += weights.get("format", 0.0)
    else:
        return reward

    # Compilation (skipped languages report compiled=True).
    if result.compiled:
        reward += weights.get("compiled", 0.0)
    else:
        return reward

    # Execution: ran on the GPU without runtime errors.
    if result.executed:
        reward += weights.get("executed", 0.0)
    else:
        return reward

    # Correctness: numerically matched the reference on every workload.
    if result.correctness:
        reward += weights.get("correctness", 0.0)
    else:
        return reward

    # Performance — PREFER the SOL score (gap toward speed-of-light, anchored at
    # human-best; the metric solswarm/KFB reward on). ``sol_score`` is in [0, 1]
    # (0.5 = match human-best, 1.0 = roofline), scaled by the performance weight.
    # Fall back to cudagym's eager-reference speedup only when the problem has no
    # SOL/human-best anchors (-1.0); correctness-only workloads also leave both -1.
    if result.sol_score >= 0.0:
        reward += weights.get("performance", 0.0) * result.sol_score
    elif result.speedup != -1.0:
        reward += normalize_performance_reward(
            result.speedup,
            scale=weights["performance"],
            clip_max=perf_reward_config["clip_max"],
            clip_min=perf_reward_config["clip_min"],
            speedup_ratio=perf_reward_config["speedup_ratio"],
        )

    return reward


def normalize_performance_reward(
    speedup: float,
    clip_max: float,
    clip_min: float,
    scale: float,
    speedup_ratio: float,
) -> float:
    """Map a speedup factor onto ``[0, scale]`` with an asymmetric log scale.

    All parameters come from the recipe (``perf_reward_config`` + the
    performance weight as ``scale``) — no defaults, so this cannot silently
    disagree with the config. ``speedup_ratio`` is the fraction of the range
    given to speedups (>=1.0x); the remainder covers slowdowns. With the
    shipped ``speedup_ratio=0.75, scale=1.0``:
      - 0.1x -> 0.0 (minimum)       - 2.0x  -> 0.48
      - 1.0x -> 0.25 (25% of range) - 10.0x -> 1.0 (maximum)
    """
    s = max(clip_min, min(speedup, clip_max))

    if s < 1.0:
        # Slowdown: map [clip_min, 1.0] -> [0, scale * (1 - speedup_ratio)].
        slowdown_scale = scale * (1 - speedup_ratio)
        val = math.log2(s) - math.log2(clip_min)
        max_val = math.log2(1.0) - math.log2(clip_min)
        return slowdown_scale * val / max_val
    else:
        # Speedup: map [1.0, clip_max] -> [scale * (1 - speedup_ratio), scale].
        offset = scale * (1 - speedup_ratio)
        speedup_scale = scale * speedup_ratio
        val = math.log2(s)
        max_val = math.log2(clip_max)
        return offset + speedup_scale * val / max_val
