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

"""GRPO dataset of SOLBench / Kernel-Factory-Bench (KFB) kernel problems.

Each *raw row* describes one problem in the SOLBench schema:
    {"definition": <JSON string of a Definition dict>,
     "workloads": <JSON string of [Workload dict, ...]>,
     "language": "triton" | "cuda_cpp" | ..., "target_hardware": "B200",
     "destination_passing_style": bool, "sol_anchors": <JSON string>}
``definition``/``workloads`` are exactly a KFB ``definition.json`` and the lines
of its ``workload.jsonl``, stored as JSON strings so the HuggingFace dataset
schema stays uniform across structurally-different problems (the data processor
parses them back). Use ``kfb_problem_to_row`` / ``write_kfb_dataset`` to build a
dataset JSONL from a KFB checkout.

``format_cuda_problem`` assigns each row a ``task_name`` (which registered env /
GPU SKU evaluates it) and passes the problem through. The per-task data
processor (``cudagym_data_processor`` in ``run_grpo_cuda.py``) later renders
the prompt and stores the problem as ``extra_env_info``.

This module imports nothing from ``cudagym`` so it can run inside DataLoader
worker subprocesses; the typed ``Definition``/``Workload`` objects are built
later, inside the env actor (``cudagym_client.parse_problem``).
"""

import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets


def _sample_task(task_to_env_config: dict[str, Any], target_hardware: Optional[str] = None) -> str:
    """Sample a task name (registered env / GPU SKU) by its config ``weight``.

    A row that pins ``target_hardware`` samples only among envs whose ``sku``
    matches (case-insensitive): routing a B200-pinned row to an h100 env would
    just trip the env's row/env mismatch guard and train as a 0-reward
    ``config_error``. No matching env is a dataset/config error — fail loudly.
    """
    tasks = list(task_to_env_config)
    if target_hardware:
        tasks = [
            t
            for t in tasks
            if str(getattr(task_to_env_config[t], "sku", "")).lower() == str(target_hardware).lower()
        ]
        if not tasks:
            raise ValueError(
                f"no configured env serves target_hardware={target_hardware!r} "
                f"(envs: {list(task_to_env_config)})"
            )
    weights = [task_to_env_config[t].weight for t in tasks]
    total = sum(weights)
    normalized = [w / total for w in weights] if total > 0 else None
    return random.choices(tasks, weights=normalized, k=1)[0]


def format_cuda_problem(
    data: dict[str, Any], task_to_env_config: dict[str, Any]
) -> dict[str, Any]:
    """Normalize one raw problem row and tag it with a sampled ``task_name``.

    Args:
        data: a raw SOLBench problem row (see module docstring).
        task_to_env_config: env-name -> ``CudaGymEvalConfig`` (carries ``weight``
            and ``sku``); the chosen task selects which env evaluates this row.
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
    """Load SOLBench problem JSONL(s) and split into train/validation.

    The problem set must hold at least ``grpo.num_prompts_per_step`` train rows
    so a step's sampler gets a full batch (the train dataloader drops the last
    partial batch). Size the JSONL accordingly and use ``grpo.max_num_epochs``
    to run more steps than one pass over the data provides.
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


# -- Kernel-Factory-Bench helpers (build a dataset JSONL from a KFB checkout) --


def load_sol_anchors(
    sol_latencies_csv: str, artifact_id: str
) -> dict[str, dict[str, float]]:
    """Per-workload SOL / human-best anchors for one problem, from a KFB latency CSV.

    Returns ``{workload_uuid: {"human_best_latency_ms", "sol_latency_ms"}}`` for every
    workload of ``artifact_id`` with a *positive* human-best latency (the essential
    anchor; ``sol_latency_ms`` may be 0 -> the SOL score degrades to a bounded
    speedup-over-human-best). Reads ``latencies_b200.csv`` / ``sol_latencies.csv``; the
    human-best column is ``human_best_latency_ms`` or ``optimized_baseline_latency_ms``.
    Loaded at dataset-build time so the anchors travel with the row (no CSV access at
    eval time). Consumed by the reward in nemo_rl/environments/atlas/cudagym_client.py.
    """
    import csv

    anchors: dict[str, dict[str, float]] = {}
    path = Path(sol_latencies_csv)
    if not path.is_file():
        return anchors
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
    """Per-workload SOL / human-best anchors from the problem's own metadata.

    KFB problem sets don't all ship a repo-level latency CSV; each problem dir
    carries ``kernel_factory_solution.json`` whose ``kernel_factory_result.
    per_workload`` rows hold the official (human-best) solution's measured
    ``latency_ms``, keyed by ``axes``. Join axes to ``workload.jsonl`` uuids::

        human_best_latency_ms = latency_ms
        sol_latency_ms        = 0.0

    ``sol_latency_ms`` is deliberately NOT derived from the file's ``sol_score``.
    That score is KFB's ANCHORED metric, ``1 / (1 + (T_k - T_sol)/(T_b - T_sol))``
    (kernel-factory-bench scripts/calculate_sol_scores.py), not a ``T_sol/T_k``
    ratio, so the speed-of-light latency is not recoverable from it — and since
    the row it describes IS the human-best run, the score is ~1 by construction
    and carries no information about ``T_sol``. Multiplying produced an arbitrary
    per-problem SOL gap that still looked well-behaved (bounded in [0, 1] and
    exactly 0.5 at the anchor), which is precisely why it would not have been
    noticed. 0.0 is the documented "no SOL anchor" value: ``sol_score`` then
    degrades to a bounded speedup-over-human-best (see cuda_kernel_utils).

    Returns ``{}`` when the file is absent, the official solution isn't marked
    correct, or no per-workload row matches — same tolerance as the CSV loader.
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
    """Read one KFB problem directory into a SOLBench dataset row.

    Args:
        problem_dir: a dir containing ``definition.json`` + ``workload.jsonl``.
        language: the cudagym ``SupportedLanguages`` value the policy must emit
            (e.g. "triton", "cuda_cpp"); KFB defines the *problem*, not the
            solution language, so it is chosen here.
        target_hardware: GPU SKU the kernel is evaluated on (KFB targets "B200").
        destination_passing_style: whether ``run`` writes outputs in-place.
        sol_latencies_csv: optional KFB latency CSV (e.g. ``data/benchmark/latencies_b200.csv``);
            when given, per-workload SOL/human-best anchors for this problem are baked in.

    ``sol_anchors`` is stored as a JSON string (not a nested dict) so the HuggingFace
    dataset schema stays uniform across structurally-different problems.
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
        # definition/workloads are stored as JSON strings (like sol_anchors) so the
        # HuggingFace dataset schema stays uniform across structurally-different KFB
        # problems -- disjoint nested axes/inputs keys would otherwise make
        # Dataset.from_json raise or silently null-fill the inferred struct. They are
        # parsed back in run_grpo_cuda's data processor.
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
    """Write a SOLBench dataset JSONL (one problem per line) from KFB dirs.

    Returns the number of rows written. Point ``data.dataset_path`` at ``out_path``.
    Pass ``sol_latencies_csv`` (e.g. ``data/benchmark/latencies_b200.csv``) to bake
    per-problem SOL/human-best anchors into each row for the SOL-score reward.

    Check the CSV has non-zero ``sol_latency_ms`` (data/sol_execbench_external does;
    data/benchmark is all zeros, so SOL degrades to speedup-over-human-best). The
    problem set also fixes the eval-server profile.
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


# Agentic task-seed prompt (a template so the words live once; `./problem/` is
# the Gym cuda_agent server's seeding convention).
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
    """One NeMo-Gym task-seed row for the agentic ``cuda_agent`` path.

    Matches the Gym cudagym resources server's expected shape (see
    3rdparty/Gym-workspace/Gym/resources_servers/cudagym/data/example.jsonl):
    ``responses_create_params.input`` = a single user turn describing the task
    (the sandbox carries the full problem files), and ``verifier_metadata`` =
    the problem the agent server seeds + the verifier scores. Unlike
    ``kfb_problem_to_row``, nested objects stay PARSED dicts/lists — Gym reads
    plain JSON rows, so there is no HuggingFace schema-uniformity constraint —
    and ``agent_ref`` is baked in (rollout routing; no ``ng_prepare_data`` pass
    needed).
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
    desc = " ".join((definition.get("description") or "").split()) or "see the problem files"
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
    """Write NeMo-Gym task-seed JSONL (agentic RL) from KFB problem dirs.

    Returns the number of rows written. Point the agentic recipe's
    ``data.train.data_path`` / ``data.validation.data_path`` at the outputs
    (grpo_cuda_agentic_*.yaml).
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
    r"""Build an RL dataset from any set of KFB problem dirs, unmodified.

    A problem dir is any directory holding ``definition.json`` +
    ``workload.jsonl`` (KFB's native layout); the row copies both verbatim and
    adds only the run-level choices KFB doesn't define (language, target
    hardware, DPS) plus SOL/human-best anchors auto-discovered from each
    problem's ``kernel_factory_solution.json`` (or a latency CSV if given).

    Examples::

        python -m nemo_rl.data.atlas_datasets.grpo_cuda_dataset \\
            ~/kfb/data/sol_execbench_external/benchmark/L1/* \\
            --out train.jsonl --language triton --target-hardware B200
        # NeMo-Gym seed rows for the agentic path:
        ... --format gym-seeds --out seeds.jsonl
    """
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__.split("\n")[0])
    parser.add_argument(
        "problems",
        nargs="+",
        help="KFB problem dirs (shell globs welcome); dirs missing "
        "definition.json/workload.jsonl are skipped with a note",
    )
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--format",
        choices=("rows", "gym-seeds"),
        default="rows",
        help="rows = single-turn SOLBench dataset; gym-seeds = agentic NeMo-Gym seeds",
    )
    parser.add_argument("--language", default="triton", help="cudagym SupportedLanguages value")
    parser.add_argument("--target-hardware", default="B200")
    parser.add_argument(
        "--destination-passing-style",
        action=argparse.BooleanOptionalAction,
        # Match kfb_problem_to_row's default (the KFB convention) — a CLI that
        # silently disagreed with the API fails every workload as the model's
        # error. --no-destination-passing-style for return-style problem sets.
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
            raw = row.get("sol_anchors") or (row.get("verifier_metadata") or {}).get("sol_anchors")
            if raw and (raw if isinstance(raw, dict) else json.loads(raw)):
                anchored += 1
    print(f"wrote {n} problems -> {args.out} ({args.format}); {anchored}/{n} rows carry sol_anchors")


if __name__ == "__main__":
    main()
