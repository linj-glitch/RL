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
"""Tests for the shared cudagym reward-observability metric aggregation.

``aggregate_kernel_metrics`` is the single source of truth for the M0 (native
``run_multi_turn_rollout``) and M1 (NeMo-Gym) metric names/semantics, so both
paths log identical W&B keys.
"""

from nemo_rl.environments.atlas.cuda_kernel_utils import aggregate_kernel_metrics


def test_empty_records_returns_empty():
    assert aggregate_kernel_metrics([]) == {}


def test_all_correct_all_anchored():
    metrics = aggregate_kernel_metrics(
        [
            {
                "correctness": True,
                "speedup": 2.0,
                "human_best_speedup": 1.5,
                "sol_score": 0.8,
            },
            {
                "correctness": True,
                "speedup": 4.0,
                "human_best_speedup": 2.5,
                "sol_score": 0.6,
            },
        ]
    )
    assert metrics["correctness_rate"] == 1.0
    assert metrics["avg_speedup_over_ref"] == 3.0
    assert metrics["avg_speedup_over_baseline"] == 2.0
    assert metrics["avg_sol_score"] == 0.7
    assert metrics["perf_ref_fallback_rate"] == 0.0


def test_perf_means_over_correct_only_and_drop_sentinels():
    metrics = aggregate_kernel_metrics(
        [
            # incorrect -> excluded from every perf mean (its 9.0s must not leak in)
            {
                "correctness": False,
                "speedup": 9.0,
                "human_best_speedup": 9.0,
                "sol_score": 9.0,
            },
            # correct but no anchor -> counts for over-ref, dropped from baseline/sol,
            # and counts as a ref fallback
            {
                "correctness": True,
                "speedup": 3.0,
                "human_best_speedup": -1.0,
                "sol_score": -1.0,
            },
            # correct + anchored
            {
                "correctness": True,
                "speedup": 5.0,
                "human_best_speedup": 2.0,
                "sol_score": 0.5,
            },
        ]
    )
    assert metrics["correctness_rate"] == 2 / 3
    assert metrics["avg_speedup_over_ref"] == 4.0  # (3 + 5) / 2 over the correct
    assert metrics["avg_speedup_over_baseline"] == 2.0  # -1 sentinel dropped
    assert metrics["avg_sol_score"] == 0.5  # -1 sentinel dropped
    assert metrics["perf_ref_fallback_rate"] == 0.5  # 1 of 2 correct had no anchor


def test_zero_sol_score_is_legit_not_a_fallback():
    # sol_score == 0.0 (degenerate anchors) is a real score, not the -1.0 sentinel.
    metrics = aggregate_kernel_metrics(
        [
            {
                "correctness": True,
                "speedup": 1.0,
                "human_best_speedup": 1.0,
                "sol_score": 0.0,
            }
        ]
    )
    assert metrics["avg_sol_score"] == 0.0
    assert metrics["perf_ref_fallback_rate"] == 0.0


def test_no_correct_kernels():
    metrics = aggregate_kernel_metrics(
        [
            {
                "correctness": False,
                "speedup": 2.0,
                "human_best_speedup": 2.0,
                "sol_score": 0.9,
            }
        ]
    )
    assert metrics["correctness_rate"] == 0.0
    assert metrics["avg_speedup_over_ref"] == 0.0
    assert metrics["avg_speedup_over_baseline"] == 0.0
    assert metrics["avg_sol_score"] == 0.0
    # No correct kernels -> the fallback-rate key is omitted (nothing to divide by).
    assert "perf_ref_fallback_rate" not in metrics


def test_missing_keys_default_to_sentinels():
    # A record with only correctness (no perf fields) -> perf means treat it as
    # unmeasured (dropped), fallback counts it (no sol anchor).
    metrics = aggregate_kernel_metrics([{"correctness": True}])
    assert metrics["correctness_rate"] == 1.0
    assert metrics["avg_speedup_over_ref"] == 0.0
    assert metrics["avg_sol_score"] == 0.0
    assert metrics["perf_ref_fallback_rate"] == 1.0
