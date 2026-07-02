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

"""Single-turn (M0) GRPO on CudaGym / SOLBench kernel problems.

End-to-end wiring (current NeMo-RL ``setup``/``grpo_train`` API):

  1. ``setup_environments`` builds one ``CudaGymEnvironment`` Ray actor per GPU
     arch in ``env.cudagym`` (a thin CudaGym HTTP client, ``num_gpus=0``).
  2. ``setup_data`` loads SOLBench problem rows (``GRPODriverDataset``), tags
     each with a sampled task (arch), and wraps them in ``AllTaskProcessedDataset``
     with ``cudagym_data_processor``.
  3. ``cudagym_data_processor`` renders the problem into a single ``user`` prompt
     (``<think>`` + fenced kernel requested), applies the chat template, stores
     ``token_ids`` + the SOLBench problem as ``extra_env_info``.
  4. GRPO generates one completion/prompt; ``CudaGymEnvironment.step`` evaluates
     it and returns the staged reward (single-turn, ``done=1``); assistant tokens
     (``<think>`` + code) train, prompt tokens are masked.

The agentic path (M1) reuses the env's evaluation+reward via a NeMo-Gym
``cuda_agent``; this file is the single-turn baseline.
"""

import argparse
import json
import os
import pprint
from typing import Any, Optional

import ray
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.atlas_datasets import GRPODriverDataset
from nemo_rl.data.datasets.processed_dataset import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.atlas.cuda_kernel_utils import (
    entry_symbol_for,
    fence_lang_for,
)
from nemo_rl.environments.atlas.cudagym_environment import CudaGymEnvironment
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

TokenizerType = PreTrainedTokenizerBase

# FQN of the env actor, used to look up its Ray runtime venv (registered in
# nemo_rl/distributed/ray_actor_environment_registry.py -> PY_EXECUTABLES.SYSTEM).
_CUDAGYM_ENV_FQN = "nemo_rl.environments.atlas.cudagym_environment.CudaGymEnvironment"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run single-turn CudaGym GRPO")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    args, overrides = parser.parse_known_args()
    return args, overrides


# ===============================================================================
#                           Prompt construction
# ===============================================================================
def _annotate_solbench_problem(
    definition: dict, destination_passing_style: bool
) -> str:
    """Render a SOLBench ``definition`` dict into a readable problem statement.

    Includes the description, the input/output tensor specs (shape + dtype), the
    axis semantics (which dims are constant vs vary per workload), the function
    signature the kernel must implement, and the reference implementation — so
    the policy knows exactly what to write and what inputs/outputs to expect.
    Operates on the raw dict (no ``cudagym`` import) since it runs in DataLoader
    workers.
    """
    lines: list[str] = []

    if definition.get("description"):
        lines.append(f"# {definition['description']}")
        lines.append("")

    input_names = list(definition.get("inputs", {}).keys())
    output_names = list(definition.get("outputs", {}).keys())

    lines.append("# Inputs:")
    for name, spec in definition.get("inputs", {}).items():
        lines.append(
            f"#   {name}: shape={spec.get('shape', [])}, dtype={spec.get('dtype', '?')}"
        )
    lines.append("# Outputs:")
    for name, spec in definition.get("outputs", {}).items():
        lines.append(
            f"#   {name}: shape={spec.get('shape', [])}, dtype={spec.get('dtype', '?')}"
        )

    # Axes: tell the model which dims are fixed (const) vs vary per workload (var/expr).
    axes = definition.get("axes", {})
    if axes:
        parts = []
        for axis_name, axis_spec in axes.items():
            if axis_spec.get("type") == "const":
                parts.append(f"{axis_name}={axis_spec.get('value')}")
            elif axis_spec.get("type") == "expr":
                parts.append(f"{axis_name}={axis_spec.get('expression')}")
            else:
                parts.append(f"{axis_name}=variable")
        lines.append(f"# Axes: {', '.join(parts)}")
    lines.append("")

    # Function signature hint: destination-passing-style passes outputs as the
    # trailing args to be written in place; otherwise the function returns them.
    if destination_passing_style:
        all_args = ", ".join(input_names + output_names)
        lines.append(f"# Function signature: run({all_args})")
        lines.append(
            f"# Outputs ({', '.join(output_names)}) are pre-allocated; write results in-place."
        )
    else:
        all_args = ", ".join(input_names)
        lines.append(f"# Function signature: run({all_args})")
        lines.append(f"# Return: {', '.join(output_names)}")
    lines.append("")

    lines.append("# Reference implementation:")
    lines.append(definition.get("reference", ""))
    return "\n".join(lines)


def cudagym_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: Optional[int],
    idx: int,
) -> DatumSpec:
    """Process one SOLBench problem row into a ``DatumSpec`` (the turn-0 prompt).

    Expects (from ``format_cuda_problem``): ``task_name``, ``definition`` (dict),
    ``workloads`` (list), ``language``, ``target_hardware``,
    ``destination_passing_style``.

    Output ``message_log`` is a single ``user`` turn holding the fully templated
    prompt + its ``token_ids`` (system prompt, if any, folded in by the chat
    template); the SOLBench problem rides in ``extra_env_info`` for the env's
    ``step``. Over-length prompts are stubbed + masked (``loss_multiplier=0``),
    matching the SFT/reference behavior.
    """
    language = datum_dict["language"]
    destination_passing_style = datum_dict.get("destination_passing_style", True)
    # definition/workloads are baked as JSON strings by the data layer (so the HF
    # dataset schema stays uniform across structurally-different problems); parse
    # them back here. Tolerate a raw dict/list for older datasets.
    definition = datum_dict["definition"]
    if isinstance(definition, str):
        definition = json.loads(definition)
    workloads = datum_dict["workloads"]
    if isinstance(workloads, str):
        workloads = json.loads(workloads)
    problem_text = _annotate_solbench_problem(definition, destination_passing_style)

    # Render the prompt template (driver_code = the problem statement; kernel_lang
    # = the fence tag the model writes; entry_function = the symbol it must define).
    user_content = problem_text
    if task_data_spec.prompt:
        user_content = task_data_spec.prompt.format(
            driver_code=problem_text,
            driver_lang="python",  # the SOLBench reference is always Python
            kernel_lang=fence_lang_for(language),
            entry_function=entry_symbol_for(language),
        )

    messages = []
    if task_data_spec.system_prompt:
        messages.append({"role": "system", "content": task_data_spec.system_prompt})
    messages.append({"role": "user", "content": user_content})
    templated: str = tokenizer.apply_chat_template(  # type: ignore[assignment]
        messages,
        tokenize=False,
        add_generation_prompt=True,  # append the assistant generation prefix
        add_special_tokens=False,
    )
    token_ids = tokenizer(templated, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ][0]
    message_log: LLMMessageLogType = [
        {"role": "user", "content": templated, "token_ids": token_ids}
    ]
    length = len(token_ids)

    # Over-length: stub the prompt to a few tokens and mask the sample out so it
    # contributes no gradient (the policy can't fit it anyway).
    loss_multiplier = 1.0
    if max_seq_length is not None and length > max_seq_length:
        for chat_message in message_log:
            chat_message["token_ids"] = chat_message["token_ids"][
                : min(4, max_seq_length // len(message_log))
            ]
        loss_multiplier = 0.0

    extra_env_info = {
        "language": language,
        "definition": definition,
        "workloads": workloads,
        "target_hardware": datum_dict.get("target_hardware"),
        "destination_passing_style": destination_passing_style,
        # Per-workload SOL/human-best anchors (parsed from the JSON string the data
        # layer baked in) — drives the SOL-score perf reward in the env's step().
        "sol_anchors": json.loads(datum_dict.get("sol_anchors") or "{}"),
    }
    return {
        "message_log": message_log,
        "length": length,
        "extra_env_info": extra_env_info,
        "loss_multiplier": loss_multiplier,
        "idx": idx,
        "task_name": datum_dict["task_name"],
    }


def setup_data(
    tokenizer: TokenizerType,
    data_config: dict[str, Any],
    grpo_config: dict[str, Any],
    task_to_env_config: dict[str, Any],
) -> tuple[AllTaskProcessedDataset, AllTaskProcessedDataset]:
    """Build the processed train/validation datasets for the CudaGym task(s)."""
    print("\n▶ Setting up data...")
    data_paths: list[str] = data_config.get("dataset_paths") or [
        data_config["dataset_path"]
    ]
    val_data_paths: list[str] = data_config.get("val_dataset_paths") or (
        [data_config["val_dataset_path"]] if data_config.get("val_dataset_path") else []
    )
    print(f"Train datasets: {data_paths}\nValidation datasets: {val_data_paths}")

    # task_to_env_config (env_name -> CudaGymEvalConfig) is passed so the dataset
    # can sample which arch/env evaluates each problem by its ``weight``.
    data = GRPODriverDataset(
        json_file_paths=data_paths,
        val_json_file_paths=val_data_paths,
        task_to_env_config=task_to_env_config,
        seed=grpo_config.get("seed", 42),
        test_size=data_config.get("test_size", 0.05),
        duplicate_train_data=data_config.get("duplicate_train_data", True),
        grpo_config=grpo_config,
    )

    default_task_spec = TaskDataSpec(
        prompt_file=data_config.get("prompt_file"),
        system_prompt_file=data_config.get("system_prompt_file"),
    )
    # Every CudaGym task shares the same processor + default prompt spec.
    task_data_processors = {
        task_name: (default_task_spec, cudagym_data_processor)
        for task_name in task_to_env_config
    }

    dataset = AllTaskProcessedDataset(
        data.formatted_ds["train"],
        tokenizer,
        default_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    val_dataset = AllTaskProcessedDataset(
        data.formatted_ds["validation"],
        tokenizer,
        default_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    return dataset, val_dataset


def setup_environments(
    env_configs: dict[str, Any],
    cluster_config: Any,
) -> tuple[dict[str, EnvironmentInterface], dict[str, Any]]:
    """Create one ``CudaGymEnvironment`` Ray actor per arch under ``env.cudagym``.

    Each actor is a thin CudaGym HTTP client (``num_gpus=0``). The CudaGym server
    URL is resolved by ``CudaGymClient.from_env()`` unless the env block sets
    ``server_url``; colocated mode injects ``CUDAGYM_UNIFIED_SERVER_URL`` into the
    environment, which is forwarded to the actor via ``runtime_env.env_vars``.
    ``cluster_config`` is accepted for parity / future colocation logic.
    """
    task_to_env: dict[str, EnvironmentInterface] = {}
    task_to_env_config: dict[str, Any] = {}

    if "cudagym" in env_configs:
        for env_name, cfg in env_configs["cudagym"].items():
            cfg = dict(cfg)
            cfg.setdefault("arch", env_name)  # default arch = env name (e.g. "b200")
            env = CudaGymEnvironment.options(  # type: ignore[attr-defined]
                num_gpus=0,  # HTTP client only; GPU work runs on the CudaGym server
                runtime_env={
                    "py_executable": get_actor_python_env(_CUDAGYM_ENV_FQN),
                    # Forward CUDAGYM_UNIFIED_SERVER_URL / CUDAGYM_AUTH_TOKEN etc.
                    "env_vars": dict(os.environ),
                },
            ).remote(cfg)
            task_to_env[env_name] = env
            task_to_env_config[env_name] = ray.get(env.get_eval_config.remote())

    if not task_to_env:
        raise ValueError(f"No 'cudagym' environment found in env config: {env_configs}")
    return task_to_env, task_to_env_config


def main() -> None:
    """Main entry point (mirrors the current ``run_grpo_*`` wiring)."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "recipes",
            "atlas",
            "grpo_cuda_b200.yaml",
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config: MasterConfig = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    # Set global RNGs before dataset building so task sampling is deterministic.
    set_seed(config.grpo["seed"])

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for GRPO"
    )
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer
    )

    print("\n▶ Setting up environments...")
    task_to_env, task_to_env_config = setup_environments(config.env, config.cluster)

    dataset, val_dataset = setup_data(
        tokenizer, config.data, config.grpo, task_to_env_config
    )

    (
        policy,
        policy_generation,
        _nemo_gym,
        cluster,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        grpo_state,
        master_config,
        _teacher_worker_groups,
        _alias_to_group_alias,
    ) = setup(config, tokenizer, dataset, val_dataset)

    grpo_train(
        policy,
        policy_generation,
        dataloader,
        val_dataloader,
        tokenizer,
        loss_fn,
        task_to_env,
        task_to_env,  # same envs for train + validation
        logger,
        checkpointer,
        grpo_state,
        master_config,
    )


if __name__ == "__main__":
    main()
