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
    {"definition": <Definition dict>, "workloads": [<Workload dict>, ...],
     "language": "triton" | "cuda_cpp" | ..., "target_hardware": "B200",
     "destination_passing_style": bool}
``definition``/``workloads`` are exactly a KFB ``definition.json`` and the lines
of its ``workload.jsonl`` — use ``kfb_problem_to_row`` / ``write_kfb_dataset`` to
build a dataset JSONL from a KFB checkout.

``format_cuda_problem`` assigns each row a ``task_name`` (which registered env /
GPU arch evaluates it, sampled by env ``weight``) and passes the problem through.
The per-task data processor (``cudagym_data_processor`` in ``run_grpo_cuda.py``)
later renders the prompt and stores the problem as ``extra_env_info``.

This module imports nothing from ``cudagym`` so it can run inside DataLoader
worker subprocesses; the typed ``Definition``/``Workload`` objects are built
later, inside the env actor (``cudagym_client.parse_problem``).
"""

import json
import math
import random
from functools import partial
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict, concatenate_datasets

# Keys every formatted row carries (HuggingFace ``datasets`` requires a uniform
# schema across rows). All problems are SOLBench-typed; there is no longer a
# driver-code modality (the old ``cudagym.envs.*`` path was removed upstream).
_SOLBENCH_FIELDS = (
    "definition",
    "workloads",
    "language",
    "target_hardware",
    "destination_passing_style",
)


def _sample_task(task_to_env_config: dict[str, Any]) -> str:
    """Sample a task name (registered env / GPU arch) by its config ``weight``."""
    tasks = list(task_to_env_config.keys())
    weights = [cfg.weight for cfg in task_to_env_config.values()]
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
            and ``arch``); the chosen task selects which env evaluates this row.
    """
    chosen_task = _sample_task(task_to_env_config)
    return {
        "task_name": chosen_task,
        "definition": data["definition"],
        "workloads": data["workloads"],
        "language": data.get("language", "triton"),
        # Fall back to the chosen env's arch when the row doesn't pin hardware.
        "target_hardware": data.get("target_hardware")
        or getattr(task_to_env_config[chosen_task], "arch", None),
        "destination_passing_style": data.get("destination_passing_style", True),
    }


def prepare_cuda_dataset(
    json_file_paths: list[str],
    val_json_file_paths: list[str],
    task_to_env_config: dict[str, Any],
    seed: int = 42,
    test_size: float = 0.05,
    duplicate_train_data: bool = False,
    grpo_config: Optional[Any] = None,
) -> DatasetDict:
    """Load SOLBench problem JSONL(s), split, and optionally duplicate to fill steps."""
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

    # Duplicate the (usually small) problem set so one epoch covers max_num_steps.
    num_prompts_per_step = grpo_config.get("num_prompts_per_step") if grpo_config else None
    max_num_steps = grpo_config.get("max_num_steps") if grpo_config else None
    if duplicate_train_data and num_prompts_per_step and max_num_steps:
        original_train_size = len(train_formatted)
        needed = num_prompts_per_step * max_num_steps
        if original_train_size > 0 and needed > original_train_size:
            factor = math.ceil(needed / original_train_size)
            print(
                f"Duplicating train data x{factor} "
                f"({original_train_size} -> {original_train_size * factor}; need {needed})."
            )
            train_formatted = concatenate_datasets([train_formatted] * factor).shuffle(
                seed=seed
            )

    # Duplicate validation up to max_val_samples (needs >= num_gpus to shard).
    max_val_samples = grpo_config.get("max_val_samples") if grpo_config else None
    if duplicate_train_data and max_val_samples and len(val_formatted) > 0:
        if len(val_formatted) < max_val_samples:
            factor = math.ceil(max_val_samples / len(val_formatted))
            val_formatted = concatenate_datasets([val_formatted] * factor).shuffle(
                seed=seed
            )

    return DatasetDict({"train": train_formatted, "validation": val_formatted})


class GRPODriverDataset:
    """Holds the formatted train/validation SOLBench problem splits.

    Mirrors the reference's container so ``run_grpo_cuda.setup_data`` can read
    ``self.formatted_ds["train"]`` / ``["validation"]``.
    """

    def __init__(
        self,
        json_file_paths: list[str],
        val_json_file_paths: list[str],
        task_to_env_config: dict[str, Any],
        seed: int = 42,
        test_size: float = 0.05,
        duplicate_train_data: bool = False,
        grpo_config: Optional[Any] = None,
    ):
        self.formatted_ds = prepare_cuda_dataset(
            json_file_paths=json_file_paths,
            val_json_file_paths=val_json_file_paths,
            task_to_env_config=task_to_env_config,
            seed=seed,
            test_size=test_size,
            duplicate_train_data=duplicate_train_data,
            grpo_config=grpo_config,
        )


# -- Kernel-Factory-Bench helpers (build a dataset JSONL from a KFB checkout) --


def kfb_problem_to_row(
    problem_dir: str,
    language: str,
    target_hardware: str = "B200",
    destination_passing_style: bool = True,
) -> dict[str, Any]:
    """Read one KFB problem directory into a SOLBench dataset row.

    Args:
        problem_dir: a dir containing ``definition.json`` + ``workload.jsonl``.
        language: the cudagym ``SupportedLanguages`` value the policy must emit
            (e.g. "triton", "cuda_cpp"); KFB defines the *problem*, not the
            solution language, so it is chosen here.
        target_hardware: GPU SKU the kernel is evaluated on (KFB targets "B200").
        destination_passing_style: whether ``run`` writes outputs in-place.
    """
    pdir = Path(problem_dir)
    definition = json.loads((pdir / "definition.json").read_text())
    workloads = [
        json.loads(line)
        for line in (pdir / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return {
        "definition": definition,
        "workloads": workloads,
        "language": language,
        "target_hardware": target_hardware,
        "destination_passing_style": destination_passing_style,
    }


def write_kfb_dataset(
    problem_dirs: list[str],
    out_path: str,
    language: str,
    target_hardware: str = "B200",
    destination_passing_style: bool = True,
) -> int:
    """Write a SOLBench dataset JSONL (one problem per line) from KFB dirs.

    Returns the number of rows written. Point ``data.dataset_path`` at ``out_path``.
    """
    rows = [
        kfb_problem_to_row(d, language, target_hardware, destination_passing_style)
        for d in problem_dirs
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)
