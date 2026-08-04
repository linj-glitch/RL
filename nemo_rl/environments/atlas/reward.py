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

Shared by the single-turn and agentic paths: the Gym cudagym resources server
(``3rdparty/Gym-workspace/Gym/resources_servers/cudagym/app.py``) vendors a
copy of this logic — keep the two in sync. A kernel that is not numerically
correct on EVERY workload earns exactly 0; a correct one earns the correctness
weight plus the performance weight scaled by the anchored SOL score (see
``get_reward``). Format/compile/execute progress is observable in the metrics
but never rewarded.
"""

import math

from .cuda_kernel_utils import KernelEvalResult


def get_reward(
    result: KernelEvalResult,
    weights: dict[str, float],
    perf_reward_config: dict[str, float],
) -> float:
    """Correctness-gated reward: 0 until the kernel is right, then pay for speed.

    A kernel that is not numerically correct on EVERY workload earns exactly 0.
    Format/compile/execution progress is observable via the result flags (and
    the aggregated metrics) but never rewarded: Triton/Python kernels have no
    ahead-of-time compile stage that can fail, so paying for "compiled" would
    reward a bare placeholder file.

    A correct kernel earns::

        weights["correctness"] + weights["performance"] * perf_term

    where ``perf_term`` is the anchored SOL score in [0, 1] (0.5 = match
    human-best, 1.0 = speed-of-light — the metric solswarm/KFB reward on) when
    the problem carries anchors. For anchor-less problems the fallback —
    log-normalized speedup over the eager reference, also mapped into [0, 1]
    before scaling — applies only when
    ``perf_reward_config["allow_speedup_fallback"]`` is true; otherwise the
    perf term is 0 and a correct kernel earns the correctness weight alone.
    """
    if not (result.formatted and result.correctness):
        return 0.0

    # Hard-indexed like normalize_performance_reward's parameters: a missing
    # weight or fallback flag is a config bug, not 0.0/True.
    reward = weights["correctness"]
    perf_weight = weights["performance"]
    # Anchored row: sol_score is the perf term (with human-best-only anchors it
    # already holds the degraded bounded speedup-over-human-best).
    if result.sol_score >= 0.0:
        reward += perf_weight * result.sol_score
    # Anchor-less row with a measured eager speedup: log-normalized fallback,
    # when the config allows it.
    elif result.speedup != -1.0 and perf_reward_config["allow_speedup_fallback"]:
        reward += normalize_performance_reward(
            result.speedup,
            scale=perf_weight,
            clip_max=perf_reward_config["clip_max"],
            clip_min=perf_reward_config["clip_min"],
            speedup_ratio=perf_reward_config["speedup_ratio"],
        )
    # Neither branch taken: no anchors and no usable speedup fallback, so the
    # correctness weight stands alone.

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
