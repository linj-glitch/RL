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

The scoring arithmetic itself (``sol_score``, ``geomean``,
``normalize_performance_reward``) lives in ``cudagym.rl`` and is the same code
the Gym cudagym resources server calls. What is mirrored rather than shared is
the correctness-gating ladder below and its counterpart in that server's
``staged_reward_from_trace``
(``3rdparty/Gym-workspace/Gym/resources_servers/cudagym/app.py``), so a change
to the gating belongs in both. A kernel that is not numerically
correct on EVERY workload earns exactly 0; a correct one earns the correctness
weight plus the performance weight scaled by the anchored SOL score (see
``get_reward``). Format/compile/execute progress is observable in the metrics
but never rewarded.
"""

from cudagym.rl import normalize_performance_reward

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
