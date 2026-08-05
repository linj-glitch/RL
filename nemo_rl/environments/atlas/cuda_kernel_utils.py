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

"""Config, result container, and metric aggregation for CudaGym kernel evaluation.

The evaluator builds typed ``Solution``/``Definition``/``Workload`` objects and
calls ``cudagym.sdk.workflows.evaluate`` — see ``cudagym_client.py``. This
module holds the three pieces only this repo needs: the recipe config model,
the per-completion result container, and the metric aggregation both rollout
paths log through.

The pieces this repo and the NeMo-Gym cudagym resources server both need — the
per-language file conventions, the GPU-SKU checks, and the reward arithmetic —
are the SDK's own ``cudagym.rl`` module; every caller imports them from there
rather than from here.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field


class CudaGymEvalConfig(BaseModel, extra="forbid"):
    """One ``env.cudagym.<name>`` recipe entry: the settings for one GPU SKU.

    The defaults mirror the shipped recipes; the single-turn and agentic paths
    consume the same fields so reward magnitudes stay comparable. ``extra`` is
    ``forbid`` because this is populated straight from user YAML: a misspelled
    ``reward_weight:`` would otherwise be dropped and the run would silently
    train against the defaults.
    """

    # GPU SKU the kernel is evaluated on. Must be a cudagym
    # ``SupportedHardware`` value (e.g. "B200"); becomes ``Solution.spec.target_hardware``
    # and selects the compile SM version server-side.
    sku: Optional[str] = None
    # Sampling weight when several cudagym envs/SKUs are registered (data mixing).
    weight: float = 1.0
    # cudagym compile/execute timeouts, in seconds.
    compilation_timeout: int = 120
    execution_timeout_per_trial: int = 60
    # Correctness-gated reward weights, consumed by ``reward.get_reward``:
    # 0 until the kernel is numerically correct on every workload, then
    # correctness + performance * perf_term (SOL score in [0,1] when the row
    # carries anchors). Progress stages (format/compiled/executed) are metrics
    # only and never rewarded: JIT-compiled languages have no compile stage
    # that can fail, so paying for "compiled" would reward placeholder files.
    reward_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "correctness": 1.0,
            "performance": 1.0,
        }
    )
    # Performance-term config. allow_speedup_fallback: for anchor-less rows,
    # whether a correct kernel may earn the perf term from log-normalized
    # speedup-over-reference (clip_* and speedup_ratio parameterize that
    # mapping, see ``cudagym.rl.normalize_performance_reward``); false -> the perf
    # term is 0 without anchors. Typed ``Any`` because the mapping is genuinely
    # mixed: the clip/ratio entries are floats and allow_speedup_fallback is a
    # bool. Declaring it ``float`` would coerce a configured ``false`` to 0.0
    # while leaving the default a real ``True``, so the value's type would
    # depend on whether the recipe spelled it out.
    perf_reward_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "clip_max": 10.0,
            "clip_min": 0.1,
            "speedup_ratio": 0.75,
            "allow_speedup_fallback": True,
        }
    )
    # Optional cudagym ``EvalConfig`` overrides (warmup, iterations, tolerances,
    # clock locking). Empty dict -> server defaults.
    benchmark_config: dict[str, Any] = Field(default_factory=dict)

    # CudaGym service location. ``server_url`` is the unified ``/compile``+``/gpu``
    # endpoint. When None, the env falls back to CUDAGYM_UNIFIED_SERVER_URL /
    # CUDAGYM_URL (+ CUDAGYM_AUTH_TOKEN) — how colocated mode injects the
    # in-allocation load-balancer address — then to ``Client.from_env()``
    # (the SDK's split CUDAGYM_{COMPILE,GPU}_SERVER_URL variables).
    server_url: Optional[str] = None
    auth_token: Optional[str] = None
    # Fail fast at env init when the endpoint's /health reports a different GPU
    # than ``sku`` — a mismatch is otherwise SILENT for Triton (kernels JIT on
    # whatever GPU serves the request and return that GPU's timings).
    verify_endpoint_sku: bool = True
    # Where the eval servers for this SKU come from. Read at submit time only
    # (``slurm/cudagym_hosting.py``, which validates it); declared here so the
    # recipe entry that carries it still validates against this model.
    hosting: Optional[dict[str, Any]] = None


@dataclass
class KernelEvalResult:
    """Outcome of evaluating one model completion against a Definition + Workloads.

    Filled stagewise by the evaluator. ``reward.get_reward`` reads
    ``formatted``+``correctness`` (the gate) and the perf signals; the other
    flags are metrics-only. A flag implies all earlier flags (a kernel that is
    ``correctness`` necessarily ``compiled`` and ``executed``). ``speedup`` is
    the mean ``speedup_factor`` over passed workloads, or -1.0 if never
    measured (so ``get_reward`` can skip the fallback perf term).
    """

    original_prompt: str = ""
    original_completion: str = ""
    formatted: bool = False  # completion parsed into a well-formed Solution
    compiled: bool = False  # all workloads compiled (or no compilation needed)
    executed: bool = False  # all workloads ran without runtime errors
    correctness: bool = False  # all workloads numerically matched the reference
    speedup: float = (
        -1.0
    )  # mean speedup over the eager reference (speedup_factor); logging + fallback
    sol_score: float = (
        -1.0
    )  # mean SOL score in [0,1] (0.5=human-best, 1.0=speed-of-light); -1 = no anchors
    human_best_speedup: float = -1.0  # geomean speedup over human-best (logging)
    runtime: float = -1.0  # mean custom-kernel latency in ms
    # Free-form diagnostics (compile/exec errors, per-workload statuses, ...);
    # included in the env observation so the agent can react to them.
    metadata: dict[str, Any] = field(default_factory=dict)


def aggregate_kernel_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-kernel evaluation records into logging metrics.

    Each record carries ``correctness`` (bool) and the perf signals ``speedup`` (over
    the eager PyTorch reference), ``human_best_speedup`` (over the human-best baseline),
    and ``sol_score`` (anchored SOL score in [0,1]); not-measured / no-anchor values use
    the ``-1.0`` sentinel. Returns (empty when ``records`` is empty):

      * ``correctness_rate`` — over all records;
      * ``avg_speedup_over_ref`` / ``avg_speedup_over_baseline`` / ``avg_sol_score`` —
        averaged over CORRECT records only, dropping ``-1.0`` sentinels so anchorless or
        incorrect samples don't drag the mean toward zero;
      * ``perf_ref_fallback_rate`` — fraction of correct records whose performance reward
        fell back to speedup-over-ref for lack of a SOL/human-best anchor (``sol_score < 0``);
      * ``submission_rate`` — fraction of records that recorded at least one scored
        submission (``n_submissions > 0``);
      * ``eval_error_rate`` — fraction of records whose ``evaluation_error`` is non-empty
        (submissions lost to eval-infrastructure failures, expired state, malformed rows).

    ``submission_rate`` and ``eval_error_rate`` read fields only the agentic path's
    verify response carries (the Gym cudagym resources server), so each is emitted
    only when at least one record has its field; single-turn records lack both and
    their metric output is unchanged.

    This is the single source of truth for these metrics: the single-turn env
    (``cudagym_environment.CudaGymEnvironment.global_post_process_and_metrics``)
    and the agentic NeMo-Gym path (``run_async_nemo_gym_rollout`` in
    ``nemo_rl/experience/rollouts.py``) both call it, so the two paths log
    identical metric names and semantics.
    """
    if not records:
        return {}

    correct_flags = [bool(r.get("correctness", False)) for r in records]
    n_correct = sum(correct_flags)
    metrics: dict[str, float] = {"correctness_rate": n_correct / len(records)}

    def _avg_correct(key: str) -> float:
        vals = [
            float(r.get(key, -1.0))
            for r, ok in zip(records, correct_flags)
            if ok and float(r.get(key, -1.0)) >= 0.0
        ]
        return sum(vals) / len(vals) if vals else 0.0

    metrics["avg_speedup_over_ref"] = _avg_correct("speedup")
    metrics["avg_speedup_over_baseline"] = _avg_correct("human_best_speedup")
    metrics["avg_sol_score"] = _avg_correct("sol_score")

    if n_correct:
        n_fallback = sum(
            1
            for r, ok in zip(records, correct_flags)
            if ok and float(r.get("sol_score", -1.0)) < 0.0
        )
        metrics["perf_ref_fallback_rate"] = n_fallback / n_correct

    # Failure-class observability for the agentic path (see the docstring):
    # without these, submissions lost to infrastructure and rollouts that never
    # submitted both just read as correctness going to 0.
    if any("n_submissions" in r for r in records):
        metrics["submission_rate"] = sum(
            1 for r in records if int(r.get("n_submissions") or 0) > 0
        ) / len(records)
    if any("evaluation_error" in r for r in records):
        metrics["eval_error_rate"] = sum(
            1 for r in records if r.get("evaluation_error")
        ) / len(records)

    return metrics
