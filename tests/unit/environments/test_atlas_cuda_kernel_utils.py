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
"""Tests for the shared cudagym metric aggregation and the anchored SOL score.

``aggregate_kernel_metrics`` is the single source of truth for the metric names
and semantics of both the single-turn env and the agentic (NeMo-Gym) path, so
the two log identical W&B keys. ``sol_score`` is the per-workload performance
term both paths reward on.
"""

import pytest
from pydantic import ValidationError

from nemo_rl.environments.atlas.cuda_kernel_utils import (
    CudaGymEvalConfig,
    aggregate_kernel_metrics,
    canonical_sku,
    sol_score,
)


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


def test_submission_and_eval_error_rates_from_agentic_records():
    # Agentic records (the Gym verify response) always carry n_submissions and
    # evaluation_error; the rates separate "never submitted" and "submissions
    # lost to infrastructure" from correctness going to 0.
    metrics = aggregate_kernel_metrics(
        [
            {
                "correctness": True,
                "speedup": 2.0,
                "human_best_speedup": 1.5,
                "sol_score": 0.8,
                "n_submissions": 2,
                "evaluation_error": None,
            },
            {
                "correctness": False,
                "speedup": -1.0,
                "human_best_speedup": -1.0,
                "sol_score": -1.0,
                "n_submissions": 0,
                "evaluation_error": "1 of 1 submissions lost to evaluation-infrastructure failures: boom",
            },
            {
                "correctness": False,
                "speedup": -1.0,
                "human_best_speedup": -1.0,
                "sol_score": -1.0,
                "n_submissions": 0,
                "evaluation_error": None,  # submitted nothing, no infra failure
            },
        ]
    )
    assert metrics["submission_rate"] == 1 / 3
    assert metrics["eval_error_rate"] == 1 / 3


def test_submission_metrics_are_omitted_for_single_turn_records():
    # Single-turn records carry neither field, so the two agentic metrics are
    # not emitted (rather than logging a misleading constant 0).
    metrics = aggregate_kernel_metrics(
        [
            {
                "correctness": True,
                "speedup": 1.0,
                "human_best_speedup": 1.0,
                "sol_score": 0.5,
            }
        ]
    )
    assert "submission_rate" not in metrics
    assert "eval_error_rate" not in metrics


def test_sol_score_anchor_points():
    assert sol_score(1.0, 1.0, 0.5) == 0.5  # matches human-best
    assert sol_score(0.5, 1.0, 0.5) == 1.0  # reaches speed-of-light
    assert sol_score(100.0, 1.0, 0.5) < 0.01  # far slower than human-best


def test_sol_score_non_finite_inputs_score_zero():
    # min/max keep their first argument when a comparison against NaN is
    # false, so without the input guard ``max(0.0, min(1.0, nan))`` returns
    # 1.0 and a NaN latency earns the maximum performance score.
    assert sol_score(float("nan"), 1.0, 0.5) == 0.0
    assert sol_score(float("inf"), 1.0, 0.5) == 0.0
    assert sol_score(float("-inf"), 1.0, 0.5) == 0.0


def test_canonical_sku_accepts_only_the_sdk_spelling():
    assert canonical_sku("B200", "src") == "B200"
    assert canonical_sku("RTX_PRO_6000", "src") == "RTX_PRO_6000"


@pytest.mark.parametrize("bad", ["b200", "rtx-pro-6000", " B200", "", None, 200])
def test_canonical_sku_rejects_near_misses(bad):
    # Each of these satisfies some case-insensitive check upstream and would
    # only fail inside build_solution, where it reads as a model error.
    with pytest.raises(ValueError, match="src"):
        canonical_sku(bad, "src")


def test_canonical_sku_names_the_value_to_write_for_a_vendor_alias():
    # GB10 is an alias the SDK resolves, but DGX_SPARK is the enum value that
    # Solution.spec.target_hardware requires.
    with pytest.raises(ValueError, match="DGX_SPARK"):
        canonical_sku("GB10", "src")
    assert sol_score(0.9, float("nan"), 0.5) == 0.0
    assert sol_score(0.9, 1.0, float("nan")) == 0.0


def test_eval_config_accepts_a_whole_recipe_entry():
    # Every key a shipped env.cudagym entry carries, including the
    # submit-time-only `hosting` block the actor itself never reads.
    config = CudaGymEvalConfig(
        sku="B200",
        weight=1.0,
        hosting={"kind": "endpoint", "endpoint": "modal/b200"},
        compilation_timeout=300,
        execution_timeout_per_trial=120,
        reward_weights={"correctness": 1.0, "performance": 1.0},
        perf_reward_config={
            "clip_max": 10.0,
            "clip_min": 0.1,
            "speedup_ratio": 0.75,
            "allow_speedup_fallback": True,
        },
        benchmark_config={"lock_clocks": True},
        verify_endpoint_sku=True,
    )
    assert config.sku == "B200"
    assert config.benchmark_config == {"lock_clocks": True}
    assert config.perf_reward_config["allow_speedup_fallback"]


def test_eval_config_rejects_a_misspelled_key():
    # A dropped `reward_weight:` would leave the run training on the default
    # weights with nothing in the logs to say so.
    with pytest.raises(ValidationError, match="reward_weight"):
        CudaGymEvalConfig(sku="B200", reward_weight={"correctness": 1.0})
