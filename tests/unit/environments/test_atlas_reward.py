# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Tests for the correctness-gated kernel reward.

The weights and the speedup-fallback flag are hard-indexed by design: a recipe
that omits one has a config bug, and reading 0.0/True in its place would
silently change the reward scale.
"""

import pytest

from nemo_rl.environments.atlas.cuda_kernel_utils import KernelEvalResult
from nemo_rl.environments.atlas.reward import get_reward

PERF_CFG = {
    "clip_max": 10.0,
    "clip_min": 0.1,
    "speedup_ratio": 0.75,
    "allow_speedup_fallback": True,
}


def _correct_result(**fields):
    result = KernelEvalResult(formatted=True, correctness=True)
    for name, value in fields.items():
        setattr(result, name, value)
    return result


def test_correct_anchored_kernel_earns_correctness_plus_scaled_sol_score():
    reward = get_reward(
        _correct_result(sol_score=0.5),
        {"correctness": 1.0, "performance": 1.0},
        PERF_CFG,
    )
    assert reward == 1.5


def test_incorrect_kernel_earns_zero():
    result = KernelEvalResult(formatted=True, correctness=False)
    assert get_reward(result, {"correctness": 1.0, "performance": 1.0}, PERF_CFG) == 0.0


def test_missing_reward_weight_is_a_config_bug():
    with pytest.raises(KeyError):
        get_reward(_correct_result(), {"performance": 1.0}, PERF_CFG)
    with pytest.raises(KeyError):
        get_reward(_correct_result(), {"correctness": 1.0}, PERF_CFG)


def test_missing_allow_speedup_fallback_is_a_config_bug():
    # Only the anchor-less-with-measured-speedup path consults the flag.
    incomplete = {k: v for k, v in PERF_CFG.items() if k != "allow_speedup_fallback"}
    with pytest.raises(KeyError):
        get_reward(
            _correct_result(speedup=2.0),
            {"correctness": 1.0, "performance": 1.0},
            incomplete,
        )
