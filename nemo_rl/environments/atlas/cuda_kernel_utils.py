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

The evaluator builds typed ``Solution``/``Definition``/``Workload`` objects and
calls ``cudagym.sdk.workflows.evaluate`` — see ``cudagym_client.py``. The
dataclasses in this module deliberately import nothing from ``cudagym`` so the
data layer (DataLoader worker subprocesses, see ``examples/run_grpo_cuda.py``)
can use the config type and the language table without the heavy dependency.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CudaGymEvalConfig:
    """Per-environment evaluation settings. One instance per registered GPU SKU.

    The defaults mirror the shipped recipes; the single-turn and agentic paths
    consume the same fields so reward magnitudes stay comparable.
    """

    # GPU SKU the kernel is evaluated on. Must be a cudagym
    # ``SupportedHardware`` value (e.g. "B200"); becomes ``Solution.spec.target_hardware``
    # and selects the compile SM version server-side. KernelFactory-Bench problems target B200.
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
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "correctness": 1.0,
            "performance": 1.0,
        }
    )
    # Performance-term config. allow_speedup_fallback: for anchor-less rows,
    # whether a correct kernel may earn the perf term from log-normalized
    # speedup-over-reference (clip_* and speedup_ratio parameterize that
    # mapping, see ``reward.normalize_performance_reward``); false -> the perf
    # term is 0 without anchors.
    perf_reward_config: dict[str, float] = field(
        default_factory=lambda: {
            "clip_max": 10.0,
            "clip_min": 0.1,
            "speedup_ratio": 0.75,
            "allow_speedup_fallback": True,
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


# ---------------------------------------------------------------------------
# Language -> (source filename, entry "<file>::run", markdown fence tag).
# ---------------------------------------------------------------------------
# Single-turn solutions are ONE file exposing a ``run`` entry. The map lives in
# this cudagym-free module so the data processor can read the fence tag and
# entry symbol inside DataLoader worker subprocesses; ``cudagym_client`` reuses
# the filename + entry point when building the typed ``Solution``. Multi-file
# C++/CUDA solutions (kernel.cu + a main.cpp pybind wrapper) do not fit in one
# fenced block and are only produced in the agentic path, where the policy
# writes files directly; for single-turn prefer a Python/Triton target (single
# file, no separate compile phase).
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


def _assert_languages_track_the_sdk() -> None:
    """Fail at import time if this table's key set drifts from ``SupportedLanguages``.

    The filenames and fence tags are this repo's convention (the SDK defines
    none), but the key set must match the SDK's language enum. Without this
    check, a language the SDK supports but the table lacks would fail one row
    at a time — as a ``KeyError`` from ``fence_lang_for`` at data time, or as
    an "unsupported language" error from ``cudagym_client.build_solution``
    recorded as the model's format error — instead of failing once at import
    with the missing names spelled out.
    """
    try:
        from cudagym.contracts.common.files import SupportedLanguages
    except ImportError:  # pragma: no cover - no-op without cudagym (data layer)
        return
    sdk = {lang.value for lang in SupportedLanguages}
    ours = set(LANGUAGE_DEFAULTS)
    if sdk - ours:
        raise RuntimeError(
            f"LANGUAGE_DEFAULTS is missing CudaGym languages {sorted(sdk - ours)}; "
            "add them (filename, entry point, fence) instead of letting rows fail as format errors"
        )
    if ours - sdk:
        raise RuntimeError(f"LANGUAGE_DEFAULTS has languages CudaGym does not support: {sorted(ours - sdk)}")


def _assert_sku_hooks_track_the_sdk() -> None:
    """Fail at import if the private SDK helpers the SKU check depends on are gone.

    An upstream rename would otherwise silently degrade every endpoint check
    to "unverifiable".
    """
    try:
        import cudagym.config.device as device
    except ImportError:  # pragma: no cover - no-op without cudagym (data layer)
        return
    for name in ("GPU_SPECS", "_hardware_match_keys", "_normalize_gpu_name"):
        if not hasattr(device, name):
            raise RuntimeError(
                f"cudagym.config.device.{name} is gone upstream; "
                "update sku_expectations/verify_health_payload to the new API"
            )


_assert_languages_track_the_sdk()
_assert_sku_hooks_track_the_sdk()


def fence_lang_for(language: str) -> str:
    """Markdown fence tag the model should write for a given cudagym language.

    Looks the language up directly, with no fallback. The import-time check
    guarantees the table covers every SDK language, so a missing key can only
    mean bad row data; raising ``KeyError`` at data time is preferable to
    rendering a prompt for a language the evaluation would then refuse.
    """
    return LANGUAGE_DEFAULTS[language][2]


def entry_symbol_for(language: str) -> str:
    """Entry function name the kernel must define (the part after ``::``)."""
    return LANGUAGE_DEFAULTS[language][1].split("::")[-1]


# ---------------------------------------------------------------------------
# Endpoint-SKU verification (the runtime half of the launcher's preflight).
# ---------------------------------------------------------------------------
# Derived from the CudaGym SDK rather than restated here: ``GPU_SPECS``
# (cudagym.config.device) carries the SM version, the compile-target qualifier,
# and the vendor name aliases for every SupportedHardware value, so a
# hand-written copy would duplicate it and drift silently. The aliases matter:
# "GB10", for example, is an alias of DGX_SPARK, not a SupportedHardware
# member of its own.
def sku_expectations(sku: str) -> Optional[tuple[tuple[str, ...], str]]:
    """``(accepted /health gpu_model substrings, expected sm prefix)`` for a SKU.

    Returns None when the name is not a SupportedHardware value (or alias), so
    callers can treat "we cannot check this" as its own outcome rather than a
    pass.
    """
    try:
        from cudagym.config.device import GPU_SPECS, _hardware_match_keys
        from cudagym.contracts.solution import SupportedHardware
    except ImportError:  # pragma: no cover - no-op without cudagym (data layer)
        return None

    # Normalize the configured name so case and hyphen/space variants still resolve.
    normalized = sku.strip().upper().replace("-", "_")
    # Resolve to a SupportedHardware member: match the enum value first, then
    # the SDK's vendor aliases.
    hardware = None
    for candidate in SupportedHardware:
        if candidate.value.upper().replace("-", "_") == normalized:
            hardware = candidate
            break
        if any(k.upper().replace("-", "_").replace(" ", "_") == normalized for k in _hardware_match_keys(candidate)):
            hardware = candidate
            break
    if hardware is None:
        return None
    # A recognized member with no GPU_SPECS entry is likewise uncheckable.
    spec = GPU_SPECS.get(hardware)
    if spec is None:
        return None
    # Accept the enum value plus every vendor alias the SDK records (GB200
    # superchip nodes report "NVIDIA GB200" while running B200 silicon). The
    # keys come back already normalized by the SDK, which is the form
    # ``verify_health_payload`` compares against.
    return tuple(_hardware_match_keys(hardware)), f"sm_{spec.sm_version}"


def verify_health_payload(payload: dict[str, Any], sku: str) -> tuple[Optional[bool], str]:
    """Compare a cudagym ``/health`` payload against a declared GPU SKU.

    Returns ``(ok, detail)``. "Unverifiable" is not a pass: it is returned as
    ``ok=None`` so a caller can distinguish "the endpoint matches" from
    "nothing was actually checked". The distinction matters because a silicon
    mismatch produces no error for Triton kernels: they JIT-compile on
    whatever GPU serves the request and report that GPU's timings.
    """
    gpu_model = payload.get("gpu_model") or ""
    sm_version = payload.get("sm_version") or ""
    if not gpu_model and not sm_version:
        return None, "unverifiable: /health reports no gpu_model/sm_version"
    expected = sku_expectations(sku)
    if expected is None:
        return None, f"unverifiable: {sku!r} is not a CudaGym SupportedHardware value"
    match_keys, sm_prefix = expected
    # Compare using the SDK's own name normalization rather than raw
    # substrings: real /health names carry vendor prefixes and spacing
    # ("NVIDIA GeForce RTX 5090"), and the normalization rules belong to the
    # SDK. Substring matching after normalization also handles the GB200 case
    # ("nvidiagb200" contains "b200", and GB200 superchip nodes do serve B200
    # kernels) without a special case here.
    if gpu_model:
        reported = _normalize_gpu(gpu_model)
        if reported is None:
            return None, f"unverifiable: cannot normalize gpu_model={gpu_model!r}"
        if not any(key and key in reported for key in match_keys):
            return (
                False,
                f"endpoint reports gpu_model={gpu_model!r}, which is not {sku}",
            )
    # The SM version is checked independently: a matching name with the wrong
    # SM version still fails.
    if sm_version and sm_prefix and not sm_version.startswith(sm_prefix):
        return (
            False,
            f"endpoint reports sm_version={sm_version!r}, expected {sm_prefix}* for {sku}",
        )
    return True, f"gpu_model={gpu_model or '?'} sm_version={sm_version or '?'}"


def _normalize_gpu(name: str) -> Optional[str]:
    """A ``/health`` gpu_model normalized the way the CudaGym SDK normalizes names."""
    try:
        from cudagym.config.device import _normalize_gpu_name
    except ImportError:  # pragma: no cover - no-op without cudagym (data layer)
        return None
    try:
        return _normalize_gpu_name(name)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# SOL score — the solswarm / Kernel-Factory-Bench (KFB) performance metric.
# ---------------------------------------------------------------------------
# Unlike cudagym's ``speedup_factor`` (speedup over the *eager* reference), the
# SOL score is anchored at the per-workload **human-best** latency (T_b) and the
# **speed-of-light** roofline (T_SOL) — both precomputed offline in KFB's
# per-suite ``latencies_b200.csv``. It is computed at reward time from
# cudagym's measured latency (T_k) plus those anchors, and it is the signal
# solswarm rewards on.
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
        fell back to speedup-over-ref for lack of a SOL/human-best anchor (``sol_score < 0``).

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

    return metrics
