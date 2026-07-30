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

    The defaults mirror the shipped recipes; the single-turn and agentic paths
    consume the same fields so reward magnitudes stay comparable.
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
    # Correctness-gated reward weights, consumed by ``reward.get_reward``:
    # 0 until the kernel is numerically correct on every workload, then
    # correctness + performance * perf_term (SOL score in [0,1] when the row
    # carries anchors). Progress rungs (format/compiled/executed) are metrics
    # only — never rewarded (JIT languages have no failable compile stage, so
    # paying for "compiled" rewarded placeholder files).
    reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "correctness": 1.0,
            "performance": 1.0,
        }
    )
    # Performance-term config. allow_speedup_fallback: for anchor-less rows,
    # whether a correct kernel may earn the perf term from log-normalized
    # speedup-over-reference (clip_*/speedup_ratio shape that mapping); false
    # -> the perf term is 0 without anchors.
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
    )  # mean speedup over the EAGER reference (cudagym; logging + fallback)
    sol_score: float = (
        -1.0
    )  # mean SOL score in [0,1] (0.5=human-best, 1.0=speed-of-light); -1 = no anchors
    human_best_speedup: float = -1.0  # geomean speedup over human-best (logging)
    runtime: float = -1.0  # mean custom-kernel latency in ms
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


def _assert_languages_track_the_sdk() -> None:
    """Fail loudly if our per-language table drifts from SupportedLanguages.

    The filenames and fence tags are genuinely ours (the SDK defines no such
    convention), but the KEY SET is the SDK's. When they diverge, an unlisted
    language reaches ``check_inline_format`` first and is reported as the
    MODEL's format error rather than as our configuration gap.
    """
    try:
        from cudagym.contracts.common.files import SupportedLanguages
    except ImportError:  # pragma: no cover - SDK always present in the atlas extra
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
    """Fail at import if the private SDK helpers the SKU check leans on are gone.

    An upstream rename would otherwise silently degrade every endpoint check
    to "unverifiable".
    """
    try:
        import cudagym.config.device as device
    except ImportError:  # pragma: no cover - the data layer runs cudagym-free
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

    Hard-indexed: the import-time guard proves the table covers every SDK
    language, so a miss can only be bad row data — fail it loudly at data time
    instead of rendering a python prompt the eval will refuse.
    """
    return LANGUAGE_DEFAULTS[language][2]


def entry_symbol_for(language: str) -> str:
    """Entry function name the kernel must define (the part after ``::``)."""
    return LANGUAGE_DEFAULTS[language][1].split("::")[-1]


# ---------------------------------------------------------------------------
# Endpoint-SKU verification (the runtime half of the launcher's preflight).
# ---------------------------------------------------------------------------
# Derived from the CudaGym SDK rather than restated here: GPU_SPECS carries the
# sm version, the compile-target qualifier and the vendor name aliases for every
# SupportedHardware value, so a hand-written table both duplicates it and goes
# stale silently. Ours had already drifted (VR100/GB300 sm versions) and listed
# "GB10", which is NOT a SupportedHardware member -- it is an alias of
# DGX_SPARK -- so an endpoint declaring it raised inside build_solution and was
# recorded as the MODEL's format error on every sample.
def sku_expectations(sku: str) -> Optional[tuple[tuple[str, ...], str]]:
    """``(accepted /health gpu_model substrings, expected sm prefix)`` for a SKU.

    Returns None when the name is not a SupportedHardware value (or alias), so
    callers can treat "we cannot check this" as its own outcome rather than a
    pass.
    """
    try:
        from cudagym.config.device import GPU_SPECS, _hardware_match_keys
        from cudagym.contracts.solution import SupportedHardware
    except ImportError:  # pragma: no cover - SDK always present in the atlas extra
        return None

    normalized = sku.strip().upper().replace("-", "_")
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
    spec = GPU_SPECS.get(hardware)
    if spec is None:
        return None
    # Accept the enum value plus every vendor alias the SDK records; GB200
    # superchip nodes report "NVIDIA GB200" while running B200 silicon, which
    # the SDK already encodes.
    # Already normalized by the SDK, which is what the comparison expects.
    return tuple(_hardware_match_keys(hardware)), f"sm_{spec.sm_version}"


def verify_health_payload(payload: dict[str, Any], sku: str) -> tuple[Optional[bool], str]:
    """Compare a cudagym ``/health`` payload against a declared GPU SKU.

    Returns ``(ok, detail)``. "Unverifiable" is NOT a pass: it comes back as
    ``ok=None`` so a caller can tell "the endpoint matches" from "nothing was
    actually checked". Reporting the latter as True is how a silicon mismatch
    stays silent -- and it is silent for Triton, which JIT-compiles on whatever
    GPU serves the request.
    """
    gpu_model = payload.get("gpu_model") or ""
    sm_version = payload.get("sm_version") or ""
    if not gpu_model and not sm_version:
        return None, "unverifiable: /health reports no gpu_model/sm_version"
    expected = sku_expectations(sku)
    if expected is None:
        return None, f"unverifiable: {sku!r} is not a CudaGym SupportedHardware value"
    match_keys, sm_prefix = expected
    # Compare with the SDK's OWN name normalization rather than raw substrings:
    # real /health names carry vendor prefixes and spacing ("NVIDIA GeForce RTX
    # 5090") that a naive test rejects, and the normalization rules belong to
    # the SDK. Substring-after-normalization also gets the GB200 case right for
    # free -- "nvidiagb200" contains "b200", and GB200 superchip nodes do serve
    # B200 kernels -- without us restating that as a special case.
    if gpu_model:
        reported = _normalize_gpu(gpu_model)
        if reported is None:
            return None, f"unverifiable: cannot normalize gpu_model={gpu_model!r}"
        if not any(key and key in reported for key in match_keys):
            return (
                False,
                f"endpoint reports gpu_model={gpu_model!r}, which is not {sku}",
            )
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
    except ImportError:  # pragma: no cover - SDK always present in the atlas extra
        return None
    try:
        return _normalize_gpu_name(name)
    except Exception:  # noqa: BLE001
        return None


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
