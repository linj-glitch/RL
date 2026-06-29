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
    """Per-environment evaluation settings. One instance per registered GPU arch.

    Reward weights / perf-normalization defaults match the reference so reward
    magnitudes are comparable across the single-turn (M0) and agentic (M1) paths.
    """

    # GPU architecture the kernel is evaluated on. Must be a cudagym
    # ``SupportedHardware`` value (e.g. "B200"); becomes ``Solution.spec.target_hardware``
    # and selects the compile SM version server-side. KFB problems target B200.
    arch: Optional[str] = None
    # Sampling weight when several cudagym envs/archs are registered (data mixing).
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
    # endpoint. When None, the env falls back to ``CudaGymClient.from_env()``
    # (reads ``CUDAGYM_UNIFIED_SERVER_URL`` / ``CUDAGYM_AUTH_TOKEN``); colocated
    # mode injects the Ray-head address here at setup time (see run_grpo_cuda.py).
    server_url: Optional[str] = None
    auth_token: Optional[str] = None


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
    speedup: float = -1.0  # mean speedup over the EAGER reference (cudagym; logging + fallback)
    sol_score: float = -1.0  # mean SOL score in [0,1] (0.5=human-best, 1.0=speed-of-light); -1 = no anchors
    human_best_speedup: float = -1.0  # geomean speedup over human-best (logging)
    runtime: float = -1.0  # mean custom-kernel latency in ms
    ref_exec_eager_time: float = -1.0  # mean reference latency in ms
    # Free-form diagnostics (compile/exec errors, per-workload statuses, ...);
    # surfaced into the env observation so the agent can react (M1).
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Language -> (source filename, entry "<file>::run", markdown fence tag).
# ---------------------------------------------------------------------------
# Single-turn (M0) solutions are ONE file exposing a ``run`` entry. This map is
# defined here -- which imports nothing from ``cudagym`` -- so the data processor
# can read the fence tag / entry symbol inside DataLoader worker subprocesses
# without the heavy dependency. ``cudagym_client`` reuses it (filename + entry)
# when building the typed ``Solution``. Multi-file C++/CUDA solutions
# (kernel.cu + main.cpp pybind) are produced agent-side in M1; for M0 prefer a
# Python/Triton target (no separate compile phase, single-file ``run``).
LANGUAGE_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "python": ("kernel.py", "kernel.py::run", "python"),
    "pytorch": ("kernel.py", "kernel.py::run", "python"),
    "triton": ("kernel.py", "kernel.py::run", "python"),
    "tilelang": ("kernel.py", "kernel.py::run", "python"),
    "cute_dsl": ("kernel.py", "kernel.py::run", "python"),
    "cutile": ("kernel.py", "kernel.py::run", "python"),
    "cudnn_frontend": ("kernel.py", "kernel.py::run", "python"),
    # C++ family: single host-compiled entry; realistic kernels are multi-file (M1).
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
    return LANGUAGE_DEFAULTS.get(language, LANGUAGE_DEFAULTS["python"])[1].split("::")[-1]


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
