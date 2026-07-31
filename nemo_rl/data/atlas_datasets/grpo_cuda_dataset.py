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

"""Datasets of KernelFactory problems for CUDA-kernel RL training.

A *KernelFactory problem* is a kernel-optimization task in the schema used by
the CudaGym evaluation service: a ``Definition`` (tensor inputs/outputs, axes,
and a Python reference implementation) plus a list of ``Workload``s (concrete
axis sizes, input specs, and tolerances). The KernelFactory-Bench (KFB) and
SOL-ExecBench problem sets use this schema; each problem is a directory
holding ``definition.json`` and ``workload.jsonl``.

This module provides two kinds of helpers:

  * Builders. ``kfb_problem_to_row`` / ``write_kfb_dataset`` turn problem
    directories into single-turn training rows; ``kfb_problem_to_gym_seed`` /
    ``write_kfb_gym_seeds`` turn them into NeMo-Gym task-seed rows for the
    agentic recipes. ``main`` exposes both from the command line.
  * Loaders. ``prepare_cuda_dataset`` loads row JSONLs and splits them into
    train/validation; ``format_cuda_problem`` tags each row with the
    ``task_name`` of the environment (GPU SKU) that will evaluate it.

A single-turn row has the fields::

    {"definition": <JSON string of a Definition dict>,
     "workloads": <JSON string of [Workload dict, ...]>,
     "language": "triton" | "cuda_cpp" | ...,
     "target_hardware": "B200",
     "destination_passing_style": bool,
     "sol_anchors": <JSON string>}

``definition`` and ``workloads`` are copied verbatim from the problem
directory but stored as JSON strings: different problems have structurally
different nested keys, and a single string column keeps the HuggingFace
dataset schema identical across rows. The data processor
(``cudagym_data_processor`` in ``examples/run_grpo_cuda.py``) parses them
back, renders the prompt, and passes the parsed problem to the environment as
``extra_env_info``. ``sol_anchors`` holds the per-workload speed-of-light and
human-best latencies the performance reward is anchored on.

This module deliberately imports nothing from the ``cudagym`` package:
dataset code runs inside DataLoader worker subprocesses, whose Python
environment is not guaranteed to have ``cudagym`` installed. The typed
``Definition``/``Workload`` objects are built later, inside the environment
actor (``parse_problem`` in ``nemo_rl/environments/atlas/cudagym_client.py``).
"""

import argparse
import csv
import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets


def _sample_task(
    task_to_env_config: dict[str, Any], target_hardware: Optional[str] = None
) -> str:
    """Sample a task name (a configured environment / GPU SKU), weighted by config ``weight``.

    A row that pins ``target_hardware`` samples only among environments whose
    ``sku`` matches it, case-insensitively. Routing a hardware-pinned row to an
    environment for different silicon would only trip that environment's
    row-vs-environment guard and score the sample as a zero-reward
    ``config_error`` (see ``nemo_rl/environments/atlas/cudagym_base.py``), so a
    pin that no configured environment serves raises here instead.
    """
    tasks = list(task_to_env_config)
    if target_hardware:
        tasks = [
            t
            for t in tasks
            if str(getattr(task_to_env_config[t], "sku", "")).lower()
            == str(target_hardware).lower()
        ]
        if not tasks:
            raise ValueError(
                f"no configured env serves target_hardware={target_hardware!r} "
                f"(envs: {list(task_to_env_config)})"
            )
    weights = [task_to_env_config[t].weight for t in tasks]
    if sum(weights) <= 0:
        raise ValueError(
            f"env weights for tasks {tasks} sum to {sum(weights)}; give at least "
            f"one env.cudagym entry a positive `weight`"
        )
    return random.choices(tasks, weights=weights, k=1)[0]


def format_cuda_problem(
    data: dict[str, Any], task_to_env_config: dict[str, Any]
) -> dict[str, Any]:
    """Normalize one raw problem row and tag it with a sampled ``task_name``.

    Args:
        data: a raw KernelFactory problem row (see the module docstring).
        task_to_env_config: mapping of environment name to ``CudaGymEvalConfig``
            (which carries ``weight`` and ``sku``). The sampled task name
            selects which environment evaluates this row.
    """
    chosen_task = _sample_task(task_to_env_config, data.get("target_hardware"))
    return {
        "task_name": chosen_task,
        "definition": data["definition"],
        "workloads": data["workloads"],
        "language": data.get("language", "triton"),
        # Fall back to the chosen env's sku when the row doesn't pin hardware.
        "target_hardware": data.get("target_hardware")
        or getattr(task_to_env_config[chosen_task], "sku", None),
        "destination_passing_style": data.get("destination_passing_style", True),
        # Per-workload SOL/human-best anchors (JSON string) for the SOL-score reward.
        "sol_anchors": data.get("sol_anchors", "{}"),
    }


def prepare_cuda_dataset(
    json_file_paths: list[str],
    val_json_file_paths: list[str],
    task_to_env_config: dict[str, Any],
    seed: int,
    test_size: float,
) -> DatasetDict:
    """Load KernelFactory problem JSONL(s) and split them into train/validation sets.

    Rows are loaded with ``Dataset.from_json`` and normalized with
    ``format_cuda_problem``. When validation files are given they are used
    as-is; otherwise the training set is split with ``test_size``.

    The training set must keep at least ``grpo.num_prompts_per_step`` rows so
    each step's sampler gets a full batch (the train dataloader drops the last
    partial batch). Size the JSONL accordingly, and raise
    ``grpo.max_num_epochs`` to run more steps than one pass over the data
    provides.
    """
    print(f"Loading datasets from {json_file_paths}...")
    original_ds = concatenate_datasets([Dataset.from_json(p) for p in json_file_paths])

    val_original_ds = None
    if val_json_file_paths:
        print(f"Loading validation datasets from {val_json_file_paths}...")
        val_original_ds = concatenate_datasets(
            [Dataset.from_json(p) for p in val_json_file_paths]
        )

    format_fn = partial(format_cuda_problem, task_to_env_config=task_to_env_config)
    formatted_ds = original_ds.map(format_fn, remove_columns=original_ds.column_names)
    val_formatted_ds = (
        val_original_ds.map(format_fn, remove_columns=val_original_ds.column_names)
        if val_original_ds is not None
        else None
    )

    if val_formatted_ds is None:
        split = formatted_ds.train_test_split(test_size=test_size, seed=seed)
        train_formatted, val_formatted = split["train"], split["test"]
    else:
        train_formatted, val_formatted = formatted_ds, val_formatted_ds

    return DatasetDict({"train": train_formatted, "validation": val_formatted})


# -- KernelFactory-Bench helpers (build dataset JSONLs from a KFB checkout) --


def load_sol_anchors(
    sol_latencies_csv: str, artifact_id: str
) -> dict[str, dict[str, float]]:
    """Load per-workload latency anchors for one problem from a KFB latency CSV.

    Reads a KFB latency CSV (for example ``latencies_b200.csv`` or
    ``sol_latencies.csv``) and returns ``{workload_uuid:
    {"human_best_latency_ms": ..., "sol_latency_ms": ...}}`` for every row of
    ``artifact_id`` whose human-best latency is positive. The human-best column
    may be named ``human_best_latency_ms`` or ``optimized_baseline_latency_ms``.
    A workload without a positive human-best latency is dropped, because the
    score cannot be anchored without one; ``sol_latency_ms`` may be 0, in which
    case the SOL score computed from it degrades to a bounded
    speedup-over-human-best (see ``sol_score`` in
    ``nemo_rl/environments/atlas/cuda_kernel_utils.py``). A nonexistent CSV
    path raises ``FileNotFoundError`` — it is always caller-provided.

    Anchors are loaded once at dataset-build time and stored on the row, so
    evaluation needs no access to the CSV. They are consumed by the reward code
    in ``nemo_rl/environments/atlas/cudagym_client.py``.
    """
    anchors: dict[str, dict[str, float]] = {}
    path = Path(sol_latencies_csv)
    if not path.is_file():
        raise FileNotFoundError(f"SOL latency CSV not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("artifact_id") != artifact_id:
                continue
            raw_hb = row.get("human_best_latency_ms") or row.get(
                "optimized_baseline_latency_ms"
            )
            try:
                human_best = float(raw_hb)
            except (TypeError, ValueError):
                continue
            if human_best <= 0.0:
                continue
            try:
                sol = float(row.get("sol_latency_ms") or 0.0)
            except (TypeError, ValueError):
                sol = 0.0
            anchors[row["workload_uuid"]] = {
                "human_best_latency_ms": human_best,
                "sol_latency_ms": sol,
            }
    return anchors


def load_sol_anchors_from_problem_dir(problem_dir: str) -> dict[str, dict[str, float]]:
    """Load per-workload latency anchors from files inside the problem directory.

    This is the fallback used when no repo-level latency CSV is supplied. A
    problem directory may carry ``kernel_factory_solution.json``, whose
    ``kernel_factory_result.per_workload`` entries record the official
    (human-best) solution's measured ``latency_ms``, keyed by ``axes``. Each
    entry is joined to a workload uuid by matching its ``axes`` against
    ``workload.jsonl`` and becomes::

        {"human_best_latency_ms": <latency_ms>, "sol_latency_ms": 0.0}

    ``sol_latency_ms`` is deliberately not derived from the file's ``sol_score``
    field. That score is KFB's anchored metric,
    ``S = 1 / (1 + (T_k - T_sol) / (T_b - T_sol))`` (see
    ``kernel-factory-bench/scripts/calculate_sol_scores.py``), not a
    ``T_sol / T_k`` ratio — and for the human-best run the file describes,
    ``T_k = T_b`` makes the score a constant that carries no information about
    ``T_sol``, so no speed-of-light latency can be recovered from it. 0.0 is
    the documented "no SOL anchor" value: the reward's SOL score then degrades
    to a bounded speedup-over-human-best (see ``sol_score`` in
    ``nemo_rl/environments/atlas/cuda_kernel_utils.py``).

    Returns ``{}`` when the file is absent, when the official solution is not
    marked correct, or when no per-workload entry matches.
    """
    pdir = Path(problem_dir)
    sol_path = pdir / "kernel_factory_solution.json"
    if not sol_path.is_file():
        return {}
    result = (json.loads(sol_path.read_text()) or {}).get("kernel_factory_result") or {}
    if not result.get("is_correct"):
        return {}
    uuid_by_axes: dict[str, str] = {}
    for line in (pdir / "workload.jsonl").read_text().splitlines():
        if line.strip():
            wl = json.loads(line)
            uuid_by_axes[json.dumps(wl["axes"], sort_keys=True)] = wl["uuid"]
    anchors: dict[str, dict[str, float]] = {}
    for pw in result.get("per_workload") or []:
        uuid = uuid_by_axes.get(json.dumps(pw.get("axes"), sort_keys=True))
        latency = float(pw.get("latency_ms") or 0.0)
        if uuid is None or latency <= 0.0:
            continue
        anchors[uuid] = {"human_best_latency_ms": latency, "sol_latency_ms": 0.0}
    return anchors


def kfb_problem_to_row(
    problem_dir: str,
    language: str,
    target_hardware: str = "B200",
    destination_passing_style: bool = True,
    sol_latencies_csv: Optional[str] = None,
) -> dict[str, Any]:
    """Read one KernelFactory-Bench problem directory into a KernelFactory-schema row.

    Args:
        problem_dir: a directory containing ``definition.json`` and
            ``workload.jsonl``.
        language: the cudagym ``SupportedLanguages`` value the policy must emit
            (for example "triton" or "cuda_cpp"). KFB defines the *problem*,
            not the solution language, so the language is chosen here.
        target_hardware: GPU SKU the kernel is evaluated on (KFB targets
            "B200").
        destination_passing_style: whether ``run`` writes outputs in place into
            trailing arguments (True) or returns them (False).
        sol_latencies_csv: optional KFB latency CSV (for example
            ``data/benchmark/latencies_b200.csv``). When given, per-workload
            SOL/human-best anchors for this problem are baked into the row;
            otherwise ``load_sol_anchors_from_problem_dir`` is tried.

    ``sol_anchors`` is stored as a JSON string (not a nested dict) so the
    HuggingFace dataset schema stays uniform across structurally different
    problems.
    """
    pdir = Path(problem_dir)
    definition = json.loads((pdir / "definition.json").read_text())
    workloads = [
        json.loads(line)
        for line in (pdir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    anchors = (
        load_sol_anchors(sol_latencies_csv, definition.get("name", pdir.name))
        if sol_latencies_csv
        else load_sol_anchors_from_problem_dir(problem_dir)
    )
    return {
        # definition/workloads stored as JSON strings (like sol_anchors): disjoint
        # nested keys across problems would make Dataset.from_json raise or
        # null-fill the inferred struct. Parsed back in run_grpo_cuda's processor.
        "definition": json.dumps(definition),
        "workloads": json.dumps(workloads),
        "language": language,
        "target_hardware": target_hardware,
        "destination_passing_style": destination_passing_style,
        "sol_anchors": json.dumps(anchors),
    }


def write_kfb_dataset(
    problem_dirs: list[str],
    out_path: str,
    language: str,
    target_hardware: str = "B200",
    destination_passing_style: bool = True,
    sol_latencies_csv: Optional[str] = None,
) -> int:
    """Write a KernelFactory problem JSONL (one row per line) from KernelFactory-Bench directories.

    Returns the number of rows written. Point ``data.dataset_path`` at
    ``out_path``. Pass ``sol_latencies_csv`` (for example
    ``data/benchmark/latencies_b200.csv``) to bake per-workload SOL/human-best
    anchors into each row for the SOL-score reward.

    Check that the CSV carries non-zero ``sol_latency_ms`` values: in a KFB
    checkout, ``data/sol_execbench_external`` ships real speed-of-light
    latencies, while ``data/benchmark``'s are all zero, so its SOL scores
    degrade to a bounded speedup-over-human-best. The problem set also decides
    which CudaGym server build must evaluate it: ``data/sol_execbench_external``
    problems need a server deployed from CudaGym's ``sol_execbench_external``
    image (selected by ``CUDAGYM_MODAL_PROFILE`` at deploy time), whose pinned
    software stack differs from the default ``kernel_factory`` image.
    """
    rows = [
        kfb_problem_to_row(
            d, language, target_hardware, destination_passing_style, sol_latencies_csv
        )
        for d in problem_dirs
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


# Prompt template for agentic task-seed rows. The Gym cuda_agent server stages
# each rollout's problem files under ./problem/ in the sandbox (see
# 3rdparty/Gym-workspace/Gym/responses_api_agents/cuda_agent/README.md).
_GYM_SEED_PROMPT = (
    "Optimize a fast GPU kernel for the `{name}` problem: {description} "
    "The full definition, workloads and reference implementation are in ./problem/. "
    "Iterate with the cudagym CLI and leave your final kernel file in the sandbox."
)


def kfb_problem_to_gym_seed(
    problem_dir: str,
    language: str,
    target_hardware: str = "B200",
    destination_passing_style: bool = True,
    sol_latencies_csv: Optional[str] = None,
    agent_name: str = "cudagym_cuda_agent",
) -> dict[str, Any]:
    """Build one NeMo-Gym task-seed row for the agentic ``cuda_agent`` recipes.

    The row matches the format the Gym cudagym resources server expects (see
    ``3rdparty/Gym-workspace/Gym/resources_servers/cudagym/data/example.jsonl``):
    ``responses_create_params.input`` is a single user turn describing the task
    (the sandbox carries the full problem files), and ``verifier_metadata`` is
    the KernelFactory problem that the agent server stages and the verifier
    scores. Two differences from ``kfb_problem_to_row``: nested objects stay
    parsed dicts/lists, because Gym reads plain JSON rows and imposes no
    HuggingFace schema-uniformity constraint; and ``agent_ref`` (which Gym
    agent server runs each rollout) is written into the row, so the JSONL is
    directly usable without a separate ``ng_prepare_data`` pass.
    """
    pdir = Path(problem_dir)
    definition = json.loads((pdir / "definition.json").read_text())
    workloads = [
        json.loads(line)
        for line in (pdir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    name = definition.get("name", pdir.name)
    anchors = (
        load_sol_anchors(sol_latencies_csv, name)
        if sol_latencies_csv
        else load_sol_anchors_from_problem_dir(problem_dir)
    )
    desc = (
        " ".join((definition.get("description") or "").split())
        or "see the problem files"
    )
    prompt = _GYM_SEED_PROMPT.format(name=name, description=desc)
    return {
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        "verifier_metadata": {
            "language": language,
            "target_hardware": target_hardware,
            "destination_passing_style": destination_passing_style,
            "definition": definition,
            "workloads": workloads,
            "sol_anchors": anchors,
        },
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
    }


def write_kfb_gym_seeds(
    problem_dirs: list[str],
    out_path: str,
    language: str,
    target_hardware: str = "B200",
    destination_passing_style: bool = True,
    sol_latencies_csv: Optional[str] = None,
    agent_name: str = "cudagym_cuda_agent",
) -> int:
    """Write a NeMo-Gym task-seed JSONL (agentic RL) from KernelFactory-Bench directories.

    Returns the number of rows written. Point the agentic recipe's
    ``data.train.data_path`` / ``data.validation.data_path`` at the outputs
    (see ``examples/configs/recipes/atlas/grpo_cuda_agentic_*.yaml``).
    """
    rows = [
        kfb_problem_to_gym_seed(
            d,
            language,
            target_hardware,
            destination_passing_style,
            sol_latencies_csv,
            agent_name,
        )
        for d in problem_dirs
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def main() -> None:
    r"""Build an RL dataset from any set of KernelFactory-Bench problem dirs, unmodified.

    A problem directory is any directory holding ``definition.json`` and
    ``workload.jsonl`` (KFB's native layout). Each row copies both files
    verbatim and adds only the run-level choices KFB does not define (language,
    target hardware, destination-passing style), plus SOL/human-best anchors
    taken from a latency CSV when one is given, or otherwise auto-discovered
    from each problem's ``kernel_factory_solution.json``.

    Examples::

        python -m nemo_rl.data.atlas_datasets.grpo_cuda_dataset \\
            ~/kfb/data/sol_execbench_external/benchmark/L1/* \\
            --out train.jsonl --language triton --target-hardware B200
        # NeMo-Gym seed rows for the agentic recipes:
        ... --format gym-seeds --out seeds.jsonl
    """
    parser = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    parser.add_argument(
        "problems",
        nargs="+",
        help="KernelFactory-Bench problem dirs (shell globs welcome); dirs missing "
        "definition.json/workload.jsonl are skipped with a note",
    )
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--format",
        choices=("rows", "gym-seeds"),
        default="rows",
        help="rows = single-turn KernelFactory problem rows; gym-seeds = agentic NeMo-Gym seeds",
    )
    parser.add_argument(
        "--language", default="triton", help="cudagym SupportedLanguages value"
    )
    parser.add_argument("--target-hardware", default="B200")
    parser.add_argument(
        "--destination-passing-style",
        action=argparse.BooleanOptionalAction,
        # Default matches kfb_problem_to_row (the KFB convention); a CLI default
        # that disagreed with it would fail every workload as the model's error.
        # Use --no-destination-passing-style for return-style problem sets.
        default=True,
        help="run() writes outputs in-place",
    )
    parser.add_argument(
        "--sol-latencies-csv",
        default=None,
        help="optional latency CSV; default = per-problem kernel_factory_solution.json",
    )
    args = parser.parse_args()

    dirs, skipped = [], []
    for p in args.problems:
        path = Path(p)
        if (path / "definition.json").is_file() and (path / "workload.jsonl").is_file():
            dirs.append(str(path))
        else:
            skipped.append(path.name)
    if skipped:
        print(f"skipped {len(skipped)} non-problem dirs: {', '.join(skipped[:8])}")
    if not dirs:
        raise SystemExit("no valid problem dirs given")

    writer = write_kfb_gym_seeds if args.format == "gym-seeds" else write_kfb_dataset
    n = writer(
        dirs,
        args.out,
        language=args.language,
        target_hardware=args.target_hardware,
        destination_passing_style=args.destination_passing_style,
        sol_latencies_csv=args.sol_latencies_csv,
    )
    anchored = 0
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            raw = row.get("sol_anchors") or (row.get("verifier_metadata") or {}).get(
                "sol_anchors"
            )
            if raw and (raw if isinstance(raw, dict) else json.loads(raw)):
                anchored += 1
    print(
        f"wrote {n} problems -> {args.out} ({args.format}); {anchored}/{n} rows carry sol_anchors"
    )
    if n and not anchored and not args.sol_latencies_csv:
        print(
            "⚠️  no row carries sol_anchors: no problem dir had kernel_factory_solution.json "
            "and no --sol-latencies-csv was given, so the performance reward degrades to "
            "speedup-over-reference (or correctness-only) for every row"
        )


if __name__ == "__main__":
    main()
