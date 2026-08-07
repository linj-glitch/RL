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

"""Glue between the atlas env and the CudaGym SDK.

Responsibilities:

  * ``parse_problem``   — KernelFactory-schema metadata dict -> typed ``Definition`` + ``Workload``s.
  * ``evaluate_solution`` — run the two-phase evaluation via ``cudagym.sdk.workflows.evaluate``
                            (it compiles solution & reference, executes, and parses) -> ``Trace``.
  * ``update_result_from_trace`` — ``Trace`` -> ``KernelEvalResult`` (the reward inputs).

The per-workload score itself (``sol_score``) and the geometric mean used to
aggregate it come from ``cudagym.rl``, so this path and the NeMo-Gym resources
server compute the same number. The owning Ray actor passes in a live
``Client``. Hard compile/execution failures are raised as
``CudaGym{Compilation,Execution}Error`` (caught by ``cudagym_base``), while
per-workload correctness/runtime outcomes are returned inside the ``Trace``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from cudagym.contracts.definition import Definition
from cudagym.contracts.eval_config import EvalConfig
from cudagym.contracts.evaluation import EvaluationStatus
from cudagym.contracts.solution import Solution
from cudagym.contracts.trace import Trace
from cudagym.contracts.workload import Workload
from cudagym.rl import geomean, sol_score
from cudagym.sdk import Client, workflows

from .cuda_kernel_utils import CudaGymEvalConfig, KernelEvalResult

# Per-workload statuses that deny a stage. ``compiled``/``executed`` flags are
# granted up to the first failed stage; they feed the metrics and the env
# observation only — the reward pays for correctness alone (see ``reward``).
_COMPILE_FAIL = {EvaluationStatus.COMPILE_ERROR}
_EXEC_FAIL = {
    EvaluationStatus.RUNTIME_ERROR,
    EvaluationStatus.TIMEOUT,
    EvaluationStatus.INVALID_REFERENCE,
}
# CUDAGRAPH_INCOMPATIBLE (numerically correct but not graph-capturable) is
# deliberately excluded: the agentic overlay bans CUDA graphs, so the status
# should not occur, and a kernel that trips it anyway earns 0.
_CORRECT_OK = {EvaluationStatus.PASSED, EvaluationStatus.CORRECTNESS_PASSED}


def parse_problem(metadata: Mapping[str, Any]) -> tuple[Definition, list[Workload]]:
    """Validate the KernelFactory problem carried in env metadata into typed models.

    Expects ``metadata["definition"]`` (a Definition dict, i.e. a problem's
    ``definition.json``) and ``metadata["workloads"]`` (a list of Workload
    dicts, i.e. the lines of its ``workload.jsonl``).
    """
    definition = Definition.model_validate(metadata["definition"])
    workloads = [Workload.model_validate(w) for w in metadata["workloads"]]
    return definition, workloads


def validate_benchmark_config(benchmark_config: dict) -> None:
    """Reject unknown ``EvalConfig`` keys instead of letting pydantic drop them.

    ``EvalConfig`` does not set ``extra="forbid"``, so a typo (``lock_clock``
    for ``lock_clocks``) is silently ignored: clocks stay unlocked while timings
    are compared against locked-clock anchors, and nothing anywhere errors.
    """
    if not benchmark_config:
        return
    unknown = set(benchmark_config) - set(EvalConfig.model_fields)
    if unknown:
        raise ValueError(
            f"unknown benchmark_config keys {sorted(unknown)}; "
            f"valid keys are {sorted(EvalConfig.model_fields)}"
        )


async def evaluate_solution(
    client: Client,
    solution: Solution,
    definition: Definition,
    workloads: list[Workload],
    eval_config: CudaGymEvalConfig,
) -> Trace:
    """Run the full two-phase evaluation and return the run-level ``Trace``.

    ``workflows.evaluate`` compiles the solution (and the reference, if it ships
    native sources), runs ``eval_driver`` on the GPU server over every workload,
    and parses the per-workload results into a ``Trace``. Raises
    ``CudaGymCompilationError`` if the solution fails to build,
    ``CudaGymExecutionError`` if the GPU job itself crashes, and
    ``RuntimeError`` if the trace is missing workloads (a truncated
    evaluation); per-workload correctness/runtime failures are returned inside
    the ``Trace`` instead.
    """
    # Empty benchmark_config means no overrides: pass None so the server
    # applies its defaults.
    config = (
        EvalConfig(**eval_config.benchmark_config)
        if eval_config.benchmark_config
        else None
    )
    # GPU timeout scales with the workload count (the SDK default is per-run).
    timeout = float(eval_config.execution_timeout_per_trial * max(1, len(workloads)))
    trace = await workflows.evaluate(
        client,
        solution=solution,
        definition=definition,
        workloads=workloads,
        config=config,
        compile_timeout=float(eval_config.compilation_timeout),
        timeout=timeout,
    )
    # The eval driver emits one workload trace per workload even for run-level
    # failures, so a shortfall means the evaluation was truncated upstream.
    # Scoring the returned subset would let a partial run earn full credit;
    # the caller records this as an evaluation (infrastructure) error.
    if len(trace.workload_traces) != len(workloads):
        raise RuntimeError(
            f"cudagym evaluation returned {len(trace.workload_traces)} workload "
            f"traces for {len(workloads)} workloads; refusing to score a "
            "partial evaluation"
        )
    return trace


def update_result_from_trace(
    trace: Trace, result: KernelEvalResult, sol_anchors: dict | None = None
) -> None:
    """Map a run-level ``Trace`` onto a ``KernelEvalResult`` in place.

    A kernel must clear a stage on *every* workload to earn it:
      * ``compiled``    — no workload reported COMPILE_ERROR (JIT path).
      * ``executed``    — and none hit RUNTIME_ERROR / TIMEOUT / INVALID_REFERENCE.
      * ``correctness`` — and every workload is PASSED / CORRECTNESS_PASSED
                          (REWARD_HACK / INCORRECT_* deny correctness).
      * ``speedup``     — mean over benchmarked workloads (``trace.summary``).
    The first failing log + per-workload statuses are stored in ``metadata`` so
    the agent can read the compiler/runtime error and revise next turn.
    """
    workload_traces = trace.workload_traces
    if not workload_traces:
        result.metadata["error"] = "no workload traces returned from evaluation"
        return

    evaluations = [wt.evaluation for wt in workload_traces]
    statuses = [e.status if e is not None else None for e in evaluations]
    # Record every workload's status up front, visible whichever stage fails below.
    result.metadata["workload_statuses"] = [
        s.value if s else "MISSING" for s in statuses
    ]

    def _first_log(predicate) -> str:
        for e in evaluations:
            if e is not None and predicate(e.status):
                return e.log
        return ""

    # Reward hacking is a hard correctness failure regardless of other stages.
    reward_hacked = any(s == EvaluationStatus.REWARD_HACK for s in statuses)
    if reward_hacked:
        result.metadata["reward_hack"] = _first_log(
            lambda s: s == EvaluationStatus.REWARD_HACK
        )

    # Stage ladder: stop at the first denied stage, so later flags stay False
    # and only that stage's error is recorded.
    result.compiled = all(s not in _COMPILE_FAIL for s in statuses)
    if not result.compiled:
        result.metadata["compile_error"] = _first_log(lambda s: s in _COMPILE_FAIL)
        return

    result.executed = all(s not in _EXEC_FAIL for s in statuses)
    if not result.executed:
        result.metadata["execution_error"] = _first_log(lambda s: s in _EXEC_FAIL)
        return

    result.correctness = (not reward_hacked) and all(s in _CORRECT_OK for s in statuses)
    if not result.correctness:
        result.metadata["correctness_error"] = _first_log(
            lambda s: s not in _CORRECT_OK
        )
        return

    # Eager-reference speedup (cudagym's own metric) — kept for logging and as
    # the FALLBACK perf signal when SOL anchors are unavailable. The truthiness
    # check also rejects the SDK's 0.0 default (an UNMEASURED reference,
    # benchmark_reference: false) — a real measured speedup is never exactly 0,
    # but 0.0 stored as "measured" drags the speedup metrics down.
    summary = trace.summary
    if summary.speedup_factor is not None and summary.speedup_factor.mean:
        result.speedup = summary.speedup_factor.mean

    # SOL score (PREFERRED — what solswarm rewards on): per workload, anchored at
    # human-best (0.5) and speed-of-light (1.0). ``sol_anchors`` maps workload uuid ->
    # {"human_best_latency_ms", "sol_latency_ms"}. Only workloads with a positive
    # human-best AND a finite positive measured latency contribute; ``sol_latency_ms`` may
    # be 0 (then ``sol_score`` degrades to a bounded speedup-over-human-best). With
    # no usable anchors sol_score stays -1 and the reward falls back to the eager
    # speedup above.
    if sol_anchors:
        scores: list[float] = []
        human_best_speedups: list[float] = []
        for workload_trace in workload_traces:
            evaluation = workload_trace.evaluation
            if evaluation is None or evaluation.performance is None:
                continue
            # WorkloadTrace.workload / Workload.uuid are required pydantic
            # fields — access them directly so an upstream rename fails loudly
            # instead of silently unmatching every anchor.
            anchor = sol_anchors.get(workload_trace.workload.uuid)
            human_best = float((anchor or {}).get("human_best_latency_ms") or 0.0)
            if not anchor or human_best <= 0.0:
                continue
            t_k = float(evaluation.performance.latency_ms)
            # 0.0 is the SDK's unmeasured default, and a NaN latency would
            # clamp to the MAXIMUM score inside sol_score (max/min keep their
            # first argument when a comparison against NaN is false); neither
            # may score, so require a finite positive measurement.
            if not (math.isfinite(t_k) and t_k > 0):
                continue
            scores.append(
                sol_score(t_k, human_best, float(anchor.get("sol_latency_ms") or 0.0))
            )
            human_best_speedups.append(human_best / t_k)
        if scores:
            result.sol_score = sum(scores) / len(scores)  # averaged across workloads
            result.metadata["sol_scores"] = scores
            # scores and human_best_speedups are appended in lockstep, so the
            # speedup list is non-empty here.
            result.human_best_speedup = geomean(human_best_speedups)
