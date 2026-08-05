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

"""Load KernelFactory-problem datasets for CUDA-kernel RL training.

A *KernelFactory problem* is a kernel-optimization task in the schema used by
the CudaGym evaluation service: a ``Definition`` (tensor inputs/outputs, axes,
and a Python reference implementation) plus a list of ``Workload``s (concrete
axis sizes, input specs, and tolerances).

This module loads pre-built row JSONLs; it does not build them. Any script may
produce rows, from any problem source, as long as each single-turn row has the
fields::

    {"definition": <JSON string of a Definition dict>,
     "workloads": <JSON string of [Workload dict, ...]>,
     "language": "triton" | "cuda_cpp" | ...,
     "target_hardware": "B200",
     "destination_passing_style": bool,
     "sol_anchors": <JSON string>}

``definition`` and ``workloads`` are stored as JSON strings: different
problems have structurally different nested keys, and a single string column
keeps the HuggingFace dataset schema identical across rows. The data processor
(``cudagym_data_processor`` in ``examples/run_grpo_cuda.py``) parses them
back, renders the prompt, and passes the parsed problem to the environment as
``extra_env_info``. The typed ``Definition``/``Workload`` objects are built
inside the environment actor (``parse_problem`` in
``nemo_rl/environments/atlas/cudagym_client.py``).

``sol_anchors`` holds per-workload latency anchors for the performance reward,
``{workload_uuid: {"human_best_latency_ms": ..., "sol_latency_ms": ...}}``.
CONTRACT: those latencies are measurements taken on the row's own
``target_hardware``. An anchor records no GPU, so nothing at evaluation time
can verify this — the dataset builder is responsible for attaching only
anchors measured on the row's GPU. Rows whose anchors were measured elsewhere
earn a plausible-looking but meaningless performance reward. ``{}`` is valid
and means "no anchors": a correct kernel then earns the correctness weight,
plus the speedup-over-reference fallback if the run times the reference
implementation (``benchmark_reference: true``).

The agentic (NeMo-Gym) seed-row format is documented by
``resources_servers/cudagym/data/example.jsonl`` in the Gym checkout; its
``verifier_metadata`` carries the same fields with nested objects left as
parsed dicts, under the same ``sol_anchors`` contract.
"""

import random
from functools import partial
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
    # Pinned row: only environments whose sku matches may serve it.
    if target_hardware:
        tasks = [
            t
            for t in tasks
            if str(task_to_env_config[t].sku).lower() == str(target_hardware).lower()
        ]
        if not tasks:
            raise ValueError(
                f"no configured env serves target_hardware={target_hardware!r} "
                f"(envs: {list(task_to_env_config)})"
            )
    # Weighted draw among the eligible environments.
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
        # Everything but target_hardware is hard-indexed: the row schema
        # requires these fields, and a silent default here would evaluate
        # the row as something the builder never declared.
        "definition": data["definition"],
        "workloads": data["workloads"],
        "language": data["language"],
        # Fall back to the chosen env's sku when the row doesn't pin hardware.
        "target_hardware": data.get("target_hardware")
        or task_to_env_config[chosen_task].sku,
        "destination_passing_style": data["destination_passing_style"],
        # Per-workload latency anchors (JSON string) for the SOL-score reward.
        "sol_anchors": data["sol_anchors"],
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

    # Normalize each row and tag it with its evaluating env; remove_columns
    # leaves only format_cuda_problem's output fields.
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
