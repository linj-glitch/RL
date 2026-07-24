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

"""Glue between the atlas env and the *current* CudaGym SDK.

Replaces the reference's per-modality ``adapters/`` + ``ExampleClient`` HTTP
code (``cudagym.envs.*`` was removed upstream). Responsibilities:

  * ``parse_problem``   — SOLBench/KFB metadata dict -> typed ``Definition`` + ``Workload``s.
  * ``build_solution``  — one extracted code block -> typed single-file ``Solution``.
  * ``evaluate_solution`` — run the two-phase eval via ``cudagym.sdk.workflows.evaluate``
                            (compile solution & reference -> execute -> parse) -> ``Trace``.
  * ``update_result_from_trace`` — ``Trace`` -> ``KernelEvalResult`` (the reward inputs).

The language -> (filename, entry, fence) map lives in ``cuda_kernel_utils``
(cudagym-free) so the data layer can read it; this module owns all the actual
``cudagym`` imports. The owning Ray actor passes in a live ``Client``;
compile/exec *hard* failures surface as ``CudaGym{Compilation,Execution}Error``
(caught by ``cudagym_base``), while per-workload correctness/runtime outcomes
ride inside the returned ``Trace``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from cudagym.contracts.common.files import SourceFile, SupportedLanguages
from cudagym.contracts.definition import Definition
from cudagym.contracts.eval_config import EvalConfig
from cudagym.contracts.evaluation import EvaluationStatus
from cudagym.contracts.solution import BuildSpec, Solution, SupportedHardware
from cudagym.contracts.trace import Trace
from cudagym.contracts.workload import Workload
from cudagym.sdk import Client, workflows

from .cuda_kernel_utils import (
    LANGUAGE_DEFAULTS,
    CudaGymEvalConfig,
    KernelEvalResult,
    geomean,
    sol_score,
)

# Per-workload statuses that deny a stage. ``compiled``/``executed`` credit is
# kept up to the first failed stage so the staged reward stays dense (see reward).
_COMPILE_FAIL = {EvaluationStatus.COMPILE_ERROR}
_EXEC_FAIL = {
    EvaluationStatus.RUNTIME_ERROR,
    EvaluationStatus.TIMEOUT,
    EvaluationStatus.INVALID_REFERENCE,
}
_CORRECT_OK = {EvaluationStatus.PASSED, EvaluationStatus.CORRECTNESS_PASSED}


def parse_problem(metadata: dict[str, Any]) -> tuple[Definition, list[Workload]]:
    """Validate the SOLBench/KFB problem carried in env metadata into typed models.

    Expects ``metadata["definition"]`` (a Definition dict, e.g. a KFB
    ``definition.json``) and ``metadata["workloads"]`` (a list of Workload dicts,
    e.g. the lines of a KFB ``workload.jsonl``).
    """
    definition = Definition.model_validate(metadata["definition"])
    workloads = [Workload.model_validate(w) for w in metadata["workloads"]]
    return definition, workloads


def build_solution(
    code: str,
    language: str,
    definition_name: str,
    target_hardware: str,
    destination_passing_style: bool,
) -> Solution:
    """Wrap one extracted code block in a typed, single-file ``Solution``.

    Args:
        code: the kernel source (one file's content).
        language: cudagym ``SupportedLanguages`` value (e.g. "triton", "cuda_cpp").
        definition_name: ``Definition.name`` this solves (links solution<->problem).
        target_hardware: GPU SKU (cudagym ``SupportedHardware`` value, e.g. "B200").
        destination_passing_style: True if ``run`` writes outputs in-place into
            trailing args, False if it returns them; must match the Definition.

    Returns:
        A frozen ``Solution`` ready for ``evaluate_solution``.

    Raises:
        ValueError / pydantic.ValidationError: unsupported language, missing
            hardware, or code that fails Solution/BuildSpec validation. The
            caller records this as a format error (reward stops at 0).
    """
    if language not in LANGUAGE_DEFAULTS:
        raise ValueError(
            f"Unsupported language {language!r}; expected one of {list(LANGUAGE_DEFAULTS)}"
        )
    if not target_hardware:
        raise ValueError(
            "target_hardware is required to build a Solution (set env sku)"
        )
    filename, entry_point, _ = LANGUAGE_DEFAULTS[language]
    # Content hash keeps the solution name (and cudagym's build cache key) stable
    # across identical completions and distinct across edits.
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
    return Solution(
        name=f"rl_{language}_{code_hash}",
        definition=definition_name,
        author="nemorl",
        spec=BuildSpec(
            languages=[SupportedLanguages(language)],
            target_hardware=[SupportedHardware(target_hardware)],
            entry_point=entry_point,
            destination_passing_style=destination_passing_style,
        ),
        sources=[SourceFile(path=filename, content=code)],
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
    ``CudaGymCompilationError`` if the solution fails to build and
    ``CudaGymExecutionError`` if the GPU job itself crashes; per-workload
    correctness/runtime failures are returned inside the ``Trace`` instead.
    """
    config = (
        EvalConfig(**eval_config.benchmark_config)
        if eval_config.benchmark_config
        else None
    )
    # GPU timeout scales with the workload count (the SDK default is per-run).
    timeout = float(eval_config.execution_timeout_per_trial * max(1, len(workloads)))
    return await workflows.evaluate(
        client,
        solution=solution,
        definition=definition,
        workloads=workloads,
        config=config,
        compile_timeout=float(eval_config.compilation_timeout),
        timeout=timeout,
    )


def update_result_from_trace(
    trace: Trace, result: KernelEvalResult, sol_anchors: dict | None = None
) -> None:
    """Map a run-level ``Trace`` onto a ``KernelEvalResult`` in place.

    A kernel must clear a stage on *every* workload to earn it:
      * ``compiled``    — no workload reported COMPILE_ERROR (JIT path).
      * ``executed``    — and none hit RUNTIME_ERROR / TIMEOUT / INVALID_REFERENCE.
      * ``correctness`` — and every workload is PASSED / CORRECTNESS_PASSED
                          (REWARD_HACK / INCORRECT_* deny correctness).
      * ``speedup``/``runtime`` — mean over benchmarked workloads (``trace.summary``).
    The first failing log + per-workload statuses are stored in ``metadata`` so
    the agent can read the compiler/runtime error and revise next turn.
    """
    workload_traces = trace.workload_traces
    if not workload_traces:
        result.metadata["error"] = "no workload traces returned from evaluation"
        return

    evaluations = [wt.evaluation for wt in workload_traces]
    statuses = [e.status if e is not None else None for e in evaluations]
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

    # Eager-reference speedup/latency (cudagym's own metric) — kept for logging
    # and as the FALLBACK perf signal when SOL anchors are unavailable.
    summary = trace.summary
    if summary.speedup_factor is not None and summary.speedup_factor.mean is not None:
        result.speedup = summary.speedup_factor.mean
    if summary.latency_ms is not None and summary.latency_ms.mean is not None:
        result.runtime = summary.latency_ms.mean

    # SOL score (PREFERRED — what solswarm/KFB reward on): per workload, anchored at
    # human-best (0.5) and speed-of-light (1.0). ``sol_anchors`` maps workload uuid ->
    # {"human_best_latency_ms", "sol_latency_ms"}. Only workloads with a positive
    # human-best contribute; ``sol_latency_ms`` may be 0 (then ``sol_score`` degrades
    # to a bounded speedup-over-human-best). No usable anchors -> sol_score stays -1
    # and the reward falls back to the eager speedup above.
    if sol_anchors:
        scores: list[float] = []
        human_best_speedups: list[float] = []
        for workload_trace in workload_traces:
            evaluation = workload_trace.evaluation
            if evaluation is None or evaluation.performance is None:
                continue
            workload = getattr(workload_trace, "workload", None)
            uuid = getattr(workload, "uuid", None)
            if uuid is None and isinstance(workload, dict):
                uuid = workload.get("uuid")
            anchor = sol_anchors.get(uuid) if uuid else None
            human_best = float((anchor or {}).get("human_best_latency_ms") or 0.0)
            if not anchor or human_best <= 0.0:
                continue
            t_k = float(evaluation.performance.latency_ms)
            scores.append(
                sol_score(t_k, human_best, float(anchor.get("sol_latency_ms") or 0.0))
            )
            if t_k > 0:
                human_best_speedups.append(human_best / t_k)
        if scores:
            result.sol_score = sum(scores) / len(
                scores
            )  # avg SOL score (KFB convention)
            result.metadata["sol_scores"] = scores
            if human_best_speedups:
                result.human_best_speedup = geomean(human_best_speedups)
