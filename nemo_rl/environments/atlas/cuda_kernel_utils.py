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

"""Config + result containers for CudaGym kernel evaluation.

These mirror the single-turn reference (``alexandery/solbench``) but target the
*current* CudaGym SDK: the evaluator no longer talks to per-modality
``cudagym.envs.*`` modules (removed upstream). It builds typed
``Solution``/``Definition``/``Workload`` objects and calls
``cudagym.sdk.workflows.evaluate`` — see ``cudagym_client.py``. These dataclasses
deliberately import nothing from ``cudagym`` so the data layer can reference the
config type without the heavy dependency.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CudaGymEvalConfig:
    """Per-environment evaluation settings. One instance per registered GPU SKU.

    Reward weights / perf-normalization defaults match the reference so reward
    magnitudes are comparable across the single-turn and agentic paths.
    """

    # GPU SKU the kernel is evaluated on. Must be a cudagym
    # ``SupportedHardware`` value (e.g. "B200"); becomes ``Solution.spec.target_hardware``
    # and selects the compile SM version server-side. KFB problems target B200.
    sku: Optional[str] = None
    # Sampling weight when several cudagym envs/SKUs are registered (data mixing).
    weight: float = 1.0
    # cudagym compile/execute timeouts, in seconds.
    compilation_timeout: int = 120
    execution_timeout_per_trial: int = 60
    # Staged reward weights, consumed by ``reward.get_reward`` (partial credit at
    # each stage: format -> compiled -> executed -> correctness -> performance).
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "format": 1,
            "compiled": 2,
            "executed": 2,
            "correctness": 4,
            "performance": 8,
        }
    )
    # Performance-reward normalization (asymmetric log-scale of the speedup).
    perf_reward_config: dict[str, float] = field(
        default_factory=lambda: {
            "clip_max": 10.0,
            "clip_min": 0.1,
            "speedup_ratio": 0.75,
        }
    )
    # Optional cudagym ``EvalConfig`` overrides (warmup, iterations, tolerances,
    # clock locking). Empty dict -> server defaults.
    benchmark_config: dict[str, Any] = field(default_factory=dict)

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


@dataclass
class KernelEvalResult:
    """Outcome of evaluating one model completion against a Definition + Workloads.

    Filled stagewise by the evaluator; ``reward.get_reward`` reads the boolean
    flags to assign partial credit. A flag implies all earlier flags (a kernel
    that is ``correctness`` necessarily ``compiled`` and ``executed``).
    ``speedup`` is the mean ``speedup_factor`` over passed workloads, or -1.0 if
    never measured (so ``get_reward`` can skip the performance term).
    """

    original_prompt: str = ""
    original_completion: str = ""
    formatted: bool = False  # completion parsed into a well-formed Solution
    compiled: bool = False  # all workloads compiled (or no compilation needed)
    executed: bool = False  # all workloads ran without runtime errors
    correctness: bool = False  # all workloads numerically matched the reference
    speedup: float = (
        -1.0
    )  # mean speedup over the EAGER reference (cudagym; logging + fallback)
    sol_score: float = (
        -1.0
    )  # mean SOL score in [0,1] (0.5=human-best, 1.0=speed-of-light); -1 = no anchors
    human_best_speedup: float = -1.0  # geomean speedup over human-best (logging)
    runtime: float = -1.0  # mean custom-kernel latency in ms
    ref_exec_eager_time: float = -1.0  # mean reference latency in ms
    # Free-form diagnostics (compile/exec errors, per-workload statuses, ...);
    # surfaced into the env observation so the agent can react.
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Language -> (source filename, entry "<file>::run", markdown fence tag).
# ---------------------------------------------------------------------------
# Single-turn solutions are ONE file exposing a ``run`` entry. This map is
# defined here -- which imports nothing from ``cudagym`` -- so the data processor
# can read the fence tag / entry symbol inside DataLoader worker subprocesses
# without the heavy dependency. ``cudagym_client`` reuses it (filename + entry)
# when building the typed ``Solution``. Multi-file C++/CUDA solutions
# (kernel.cu + main.cpp pybind) are produced agent-side in the agentic path; for single-turn prefer a
# Python/Triton target (no separate compile phase, single-file ``run``).
LANGUAGE_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "python": ("kernel.py", "kernel.py::run", "python"),
    "pytorch": ("kernel.py", "kernel.py::run", "python"),
    "triton": ("kernel.py", "kernel.py::run", "python"),
    "tilelang": ("kernel.py", "kernel.py::run", "python"),
    "cute_dsl": ("kernel.py", "kernel.py::run", "python"),
    "cutile": ("kernel.py", "kernel.py::run", "python"),
    "cudnn_frontend": ("kernel.py", "kernel.py::run", "python"),
    # C++ family: single host-compiled entry; realistic kernels are multi-file (agentic path).
    "cuda_cpp": ("main.cpp", "main.cpp::run", "cpp"),
    "cutlass": ("main.cu", "main.cu::run", "cpp"),
    "cudnn": ("main.cpp", "main.cpp::run", "cpp"),
    "cublas": ("main.cpp", "main.cpp::run", "cpp"),
}


def fence_lang_for(language: str) -> str:
    """Markdown fence tag the model should write for a given cudagym language."""
    return LANGUAGE_DEFAULTS.get(language, LANGUAGE_DEFAULTS["python"])[2]


def entry_symbol_for(language: str) -> str:
    """Entry function name the kernel must define (the part after ``::``)."""
    return LANGUAGE_DEFAULTS.get(language, LANGUAGE_DEFAULTS["python"])[1].split("::")[
        -1
    ]


# ---------------------------------------------------------------------------
# Endpoint-SKU verification (the runtime half of the launcher's preflight).
# ---------------------------------------------------------------------------
# SKU -> (accepted /health gpu_model substrings, expected sm_version prefix).
# B200 accepts GB200: GB200 superchip nodes report "NVIDIA GB200" but run B200
# silicon (sm_100). Kept in sync with slurm/cudagym_hosting.py and the Gym
# cudagym resources server (intentional small duplication across packages).
SKU_EXPECTATIONS: dict[str, tuple[tuple[str, ...], Optional[str]]] = {
    "B200": (("B200", "GB200"), "sm_100"),
    "H100": (("H100",), "sm_90"),
    "H200": (("H200",), "sm_90"),
    "GB10": (("GB10",), None),
    "GB300": (("GB300",), "sm_103"),
    "VR100": (("VR100",), None),
}


def verify_health_payload(payload: dict[str, Any], sku: str) -> tuple[bool, str]:
    """Compare a cudagym ``/health`` payload against a declared GPU SKU.

    Returns ``(ok, detail)``. Unverifiable payloads (no gpu fields — e.g. a
    compile-only responder) and SKUs without recorded expectations are ``ok``
    with an explanatory detail so callers can warn instead of fail.
    """
    gpu_model = payload.get("gpu_model") or ""
    sm_version = payload.get("sm_version") or ""
    if not gpu_model and not sm_version:
        return True, "unverifiable: /health reports no gpu_model/sm_version"
    expected = SKU_EXPECTATIONS.get(sku.upper())
    if expected is None:
        return True, f"unverifiable: no expectations recorded for sku {sku}"
    models, sm_prefix = expected
    if gpu_model and not any(m in gpu_model for m in models):
        return (
            False,
            f"endpoint reports gpu_model={gpu_model!r}, expected one of {models} for {sku}",
        )
    if sm_version and sm_prefix and not sm_version.startswith(sm_prefix):
        return (
            False,
            f"endpoint reports sm_version={sm_version!r}, expected {sm_prefix}* for {sku}",
        )
    return True, f"gpu_model={gpu_model or '?'} sm_version={sm_version or '?'}"


# ---------------------------------------------------------------------------
# SOL score — the solswarm / Kernel-Factory-Bench performance metric.
# ---------------------------------------------------------------------------
# Unlike cudagym's ``speedup_factor`` (speedup over the *eager* reference), the
# SOL score is anchored at the per-workload **human-best** latency (T_b) and the
# **speed-of-light** roofline (T_SOL) — both precomputed offline in KFB's
# latencies_b200.csv. We compute it at reward time from cudagym's measured
# latency (T_k) + those anchors. This is the signal solswarm rewards on.
def sol_score(latency_ms: float, human_best_ms: float, sol_ms: float) -> float:
    """Anchored Speed-Of-Light score in [0, 1] for one workload.

    ``S = 1 / (1 + (T_k - T_SOL) / (T_b - T_SOL))``, clamped to [0, 1]:
      * ``S = 0.5`` when the kernel matches human-best (T_k = T_b),
      * ``S = 1.0`` when it reaches speed-of-light (T_k = T_SOL),
      * ``S -> 0`` as it gets slower than human-best.
    Mirrors ``kernel-factory-bench/scripts/calculate_sol_scores.py`` (clamped
    here so the RL reward stays bounded).
    """
    gap = human_best_ms - sol_ms
    if gap <= 0:  # degenerate anchors (human-best already at/under SOL): pass/fail
        s = 1.0 if latency_ms <= sol_ms else 0.0
    else:
        s = 1.0 / (1.0 + (latency_ms - sol_ms) / gap)
    return max(0.0, min(1.0, s))


def geomean(values: list[float]) -> float:
    """Geometric mean of positive values (0.0 if none are positive)."""
    import math

    positives = [v for v in values if v > 0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def aggregate_kernel_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-kernel eval records into observability metrics (shared by the single-turn and agentic paths).

    Each record carries ``correctness`` (bool) and the perf signals ``speedup`` (over
    the eager PyTorch reference), ``human_best_speedup`` (over the human-best baseline),
    and ``sol_score`` (anchored SOL score in [0,1]); not-measured / no-anchor values use
    the ``-1.0`` sentinel. Returns (empty when ``records`` is empty):

      * ``correctness_rate`` — over all records;
      * ``avg_speedup_over_ref`` / ``avg_speedup_over_baseline`` / ``avg_sol_score`` —
        averaged over CORRECT records only, dropping ``-1.0`` sentinels so anchorless or
        incorrect samples don't drag the mean toward zero;
      * ``perf_ref_fallback_rate`` — fraction of correct records whose performance reward
        fell back to speedup-over-ref for lack of a SOL/human-best anchor (``sol_score < 0``).

    This is the single source of truth for the reward-observability metrics so the
    single-turn (native ``run_multi_turn_rollout``) and agentic (NeMo-Gym) paths log
    identical names/semantics.
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

    return metrics
