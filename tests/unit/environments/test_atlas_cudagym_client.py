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
"""Tests for the CudaGym SDK glue: trace-to-result mapping and evaluation guards.

Kept semantically in sync with the Gym resources server twin
(``3rdparty/Gym-workspace/Gym/resources_servers/cudagym/app.py``), which has
matching tests for the same two behaviors: unmeasured (0.0) latencies must not
score, and a trace missing workloads must raise instead of earning credit for
the returned subset.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("cudagym")

from cudagym.contracts.evaluation import EvaluationStatus  # noqa: E402

from nemo_rl.environments.atlas import cudagym_client  # noqa: E402
from nemo_rl.environments.atlas.cuda_kernel_utils import (  # noqa: E402
    CudaGymEvalConfig,
    KernelEvalResult,
)


def _passed_trace(latencies):
    """A stand-in Trace: one PASSED workload trace per latency, uuids w0, w1, ..."""
    return SimpleNamespace(
        workload_traces=[
            SimpleNamespace(
                evaluation=SimpleNamespace(
                    status=EvaluationStatus.PASSED,
                    performance=SimpleNamespace(latency_ms=latency),
                    log="",
                ),
                workload=SimpleNamespace(uuid=f"w{i}"),
            )
            for i, latency in enumerate(latencies)
        ],
        summary=SimpleNamespace(speedup_factor=None, latency_ms=None),
    )


# --- SOL-score input guard: latency 0.0 is unmeasured, not speed-of-light ---


def test_zero_latency_workloads_are_skipped_by_the_sol_score():
    anchors = {
        "w0": {"human_best_latency_ms": 1.0, "sol_latency_ms": 0.0},
        "w1": {"human_best_latency_ms": 1.0, "sol_latency_ms": 0.0},
    }
    result = KernelEvalResult()
    # w0 reports the SDK's unmeasured 0.0 latency and must not contribute; w1
    # matches human-best exactly, so the mean SOL score is exactly 0.5.
    cudagym_client.update_result_from_trace(
        _passed_trace([0.0, 1.0]), result, sol_anchors=anchors
    )
    assert result.correctness is True
    assert result.sol_score == 0.5
    assert result.metadata["sol_scores"] == [0.5]
    assert result.human_best_speedup == 1.0


def test_all_latencies_unmeasured_leaves_the_no_anchor_sentinel():
    anchors = {"w0": {"human_best_latency_ms": 1.0, "sol_latency_ms": 0.0}}
    result = KernelEvalResult()
    cudagym_client.update_result_from_trace(
        _passed_trace([0.0]), result, sol_anchors=anchors
    )
    assert result.correctness is True
    assert result.sol_score == -1.0
    assert "sol_scores" not in result.metadata


# --- evaluation guard: a trace missing workloads is an infrastructure error ---


def test_evaluate_solution_raises_on_a_partial_trace(monkeypatch):
    async def fake_evaluate(client, **kwargs):
        return _passed_trace([1.0])  # one trace for two workloads

    monkeypatch.setattr(cudagym_client.workflows, "evaluate", fake_evaluate)
    with pytest.raises(RuntimeError, match="partial evaluation"):
        asyncio.run(
            cudagym_client.evaluate_solution(
                client=None,
                solution=None,
                definition=None,
                workloads=[object(), object()],
                eval_config=CudaGymEvalConfig(),
            )
        )


def test_evaluate_solution_returns_a_complete_trace_unchanged(monkeypatch):
    trace = _passed_trace([1.0, 2.0])

    async def fake_evaluate(client, **kwargs):
        return trace

    monkeypatch.setattr(cudagym_client.workflows, "evaluate", fake_evaluate)
    out = asyncio.run(
        cudagym_client.evaluate_solution(
            client=None,
            solution=None,
            definition=None,
            workloads=[object(), object()],
            eval_config=CudaGymEvalConfig(),
        )
    )
    assert out is trace
