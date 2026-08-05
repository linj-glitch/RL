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

"""Single-turn GRPO on CudaGym / KernelFactory problems.

End-to-end wiring:

  1. ``setup_environments`` builds one ``CudaGymEnvironment`` Ray actor per GPU
     SKU in ``env.cudagym`` (a thin CudaGym HTTP client, ``num_gpus=0``).
  2. ``setup_data`` loads KernelFactory problem rows (``prepare_cuda_dataset``), tags
     each with a sampled task (SKU), and wraps them in ``AllTaskProcessedDataset``
     with ``cudagym_data_processor``.
  3. ``cudagym_data_processor`` renders the problem into a single ``user`` prompt
     (``<think>`` + fenced kernel requested), applies the chat template, stores
     ``token_ids`` + the KernelFactory problem as ``extra_env_info``.
  4. Each episode is one model turn: GRPO samples a completion per (repeated)
     prompt, and ``CudaGymEnvironment.step`` evaluates it and returns the
     correctness-gated reward (``done=1``). Assistant tokens (``<think>`` + code)
     train; prompt tokens are masked.

The agentic path reuses the env's evaluation+reward via a NeMo-Gym
``cuda_agent``; this file is the single-turn baseline.
"""

import argparse
import json
import os
import pprint
from typing import Any, Optional

import ray
from cudagym.rl import entry_symbol_for, fence_lang_for
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.atlas_datasets import prepare_cuda_dataset
from nemo_rl.data.datasets.processed_dataset import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType, TaskDataSpec
from nemo_rl.distributed.ray_actor_environment_registry import get_actor_python_env
from nemo_rl.distributed.virtual_cluster import init_ray
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
# Problem-statement template. All problem-statement wording lives in this one
# string; the code below only computes the values. The surrounding instruction
# text lives in the prompt_file template (examples/prompts/cudagym.txt), whose
# {driver_code} slot this renders.
_PROBLEM_TEMPLATE = """\
{description}# Inputs:
{input_lines}
# Outputs:
{output_lines}
{axes_line}
# Function signature: {entry_function}({args})
# {output_convention}

# Reference implementation:
{reference}"""


def _tensor_lines(specs: dict) -> str:
    """Render one ``#   <name>: shape=..., dtype=...`` line per tensor spec."""
    # Hard-indexed on purpose: a definition missing shape/dtype should fail the
    # row loudly at data time, not render a '?' prompt the model can't solve.
    return "\n".join(
        f"#   {name}: shape={spec['shape']}, dtype={spec['dtype']}"
        for name, spec in specs.items()
    )


def _axis_parts(axes: dict) -> list[str]:
    """Render one ``name=value`` part per axis.

    Const axes show their value and expr axes their expression; every other
    axis varies per workload and renders as ``name=variable``.
    """
    parts = []
    for axis_name, axis_spec in axes.items():
        if axis_spec.get("type") == "const":
            parts.append(f"{axis_name}={axis_spec.get('value')}")
        elif axis_spec.get("type") == "expr":
            parts.append(f"{axis_name}={axis_spec.get('expression')}")
        else:
            parts.append(f"{axis_name}=variable")
    return parts


def _annotate_kernelfactory_problem(
    definition: dict, destination_passing_style: bool, entry_function: str
) -> str:
    """Render a KernelFactory-schema ``definition`` dict into a readable problem statement.

    ``entry_function`` comes from the same ``entry_symbol_for`` lookup the outer
    prompt template uses, so the signature line can never contradict the header.
    Reads the raw dict rather than a typed ``Definition``: the row stores the
    definition as JSON, and rendering it needs no validation.
    """
    input_names = list(definition.get("inputs", {}).keys())
    output_names = list(definition.get("outputs", {}).keys())
    axes = definition.get("axes", {})
    # Destination-passing-style passes outputs as the trailing args to be
    # written in place; otherwise the function returns them.
    if destination_passing_style:
        args = ", ".join(input_names + output_names)
        output_convention = f"Outputs ({', '.join(output_names)}) are pre-allocated; write results in-place."
    else:
        args = ", ".join(input_names)
        output_convention = f"Return: {', '.join(output_names)}"
    return _PROBLEM_TEMPLATE.format(
        description=f"# {definition['description']}\n\n"
        if definition.get("description")
        else "",
        input_lines=_tensor_lines(definition.get("inputs", {})),
        output_lines=_tensor_lines(definition.get("outputs", {})),
        axes_line=f"# Axes: {', '.join(_axis_parts(axes))}\n" if axes else "",
        entry_function=entry_function,
        args=args,
        output_convention=output_convention,
        reference=definition["reference"],
    )


def _parse_json_field(datum_dict: dict[str, Any], key: str) -> Any:
    """Parse a dataset field stored as a JSON string.

    The dataset builder (``write_kfb_dataset`` in
    ``nemo_rl/data/atlas_datasets/grpo_cuda_dataset.py``) stores
    ``definition``/``workloads``/``sol_anchors`` as JSON strings so the
    HuggingFace dataset schema stays uniform across structurally-different
    problems. Any other type means the row came from a different builder, so
    raise a ``TypeError`` that names the expected format.
    """
    value = datum_dict[key]
    if not isinstance(value, str):
        raise TypeError(
            f"dataset field {key!r} must be a JSON string, got "
            f"{type(value).__name__}; rebuild the dataset with "
            "nemo_rl.data.atlas_datasets.grpo_cuda_dataset.write_kfb_dataset "
            "(it stores this field as a JSON string)"
        )
    return json.loads(value)


def cudagym_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: Optional[int],
    idx: int,
) -> DatumSpec:
    """Process one KernelFactory problem row into a ``DatumSpec`` (the turn-0 prompt).

    Expects (from ``format_cuda_problem``): ``task_name``, ``definition`` /
    ``workloads`` (JSON strings), ``language``, ``target_hardware``,
    ``destination_passing_style``.

    Output ``message_log`` is a single ``user`` turn holding the fully templated
    prompt + its ``token_ids`` (system prompt, if any, folded in by the chat
    template); the KernelFactory problem rides in ``extra_env_info`` for the env's
    ``step``. Over-length prompts are stubbed + masked (``loss_multiplier=0``),
    matching the built-in processors in ``nemo_rl/data/processors.py``.
    """
    language = datum_dict["language"]
    # Hard-indexed: format_cuda_problem always writes the key, and a wrong
    # guess would render a prompt whose output convention the evaluation
    # then rejects on every workload.
    destination_passing_style = datum_dict["destination_passing_style"]
    # definition/workloads are baked as JSON strings by the data layer; parse
    # them back here.
    definition = _parse_json_field(datum_dict, "definition")
    workloads = _parse_json_field(datum_dict, "workloads")
    # Render the definition into the human-readable problem statement.
    entry_function = entry_symbol_for(language)
    problem_text = _annotate_kernelfactory_problem(
        definition, destination_passing_style, entry_function
    )

    # Render the prompt template (driver_code = the problem statement; kernel_lang
    # = the fence tag the model writes; entry_function = the symbol it must define;
    # gpu_sku/language = the eval target, stated explicitly so the model knows what
    # hardware and kernel dialect it is writing for).
    user_content = problem_text
    if task_data_spec.prompt:
        user_content = task_data_spec.prompt.format(
            driver_code=problem_text,
            driver_lang="python",  # the KernelFactory-schema reference is always Python
            kernel_lang=fence_lang_for(language),
            entry_function=entry_function,
            gpu_sku=datum_dict["target_hardware"],
            language=language,
        )

    # Apply the chat template and tokenize; the fully templated prompt travels
    # as a single user turn.
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

    # Per-workload SOL/human-best anchors: baked as a JSON string by the data
    # layer (like definition/workloads above) — drives the SOL-score perf
    # reward in the env's step(). An absent/empty field means "no anchors".
    sol_anchors = (
        _parse_json_field(datum_dict, "sol_anchors")
        if datum_dict.get("sol_anchors")
        else {}
    )
    extra_env_info = {
        "language": language,
        "definition": definition,
        "workloads": workloads,
        "target_hardware": datum_dict.get("target_hardware"),
        "destination_passing_style": destination_passing_style,
        "sol_anchors": sol_anchors,
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
    data_paths: list[str] = [data_config["dataset_path"]]
    val_data_paths: list[str] = (
        [data_config["val_dataset_path"]] if data_config.get("val_dataset_path") else []
    )
    print(f"Train datasets: {data_paths}\nValidation datasets: {val_data_paths}")

    # task_to_env_config (env_name -> CudaGymEvalConfig) is passed so the dataset
    # can route each problem to an env: hardware-pinned rows go to a matching
    # SKU, unpinned rows sample by env ``weight``.
    formatted_ds = prepare_cuda_dataset(
        json_file_paths=data_paths,
        val_json_file_paths=val_data_paths,
        task_to_env_config=task_to_env_config,
        seed=grpo_config["seed"],
        test_size=data_config["test_size"],
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
        formatted_ds["train"],
        tokenizer,
        default_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    val_dataset = AllTaskProcessedDataset(
        formatted_ds["validation"],
        tokenizer,
        default_task_spec,
        task_data_processors,
        max_seq_length=data_config["max_input_seq_length"],
    )
    return dataset, val_dataset


def setup_environments(
    env_configs: dict[str, Any],
) -> tuple[dict[str, EnvironmentInterface], dict[str, Any]]:
    """Create one ``CudaGymEnvironment`` Ray actor per GPU SKU under ``env.cudagym``.

    Each actor is a thin CudaGym HTTP client (``num_gpus=0``). The CudaGym server
    URL is resolved from ``CUDAGYM_UNIFIED_SERVER_URL`` unless the env block sets
    ``server_url``; the driver's environment (URL, auth token, Modal proxy pair)
    reaches the actor through the JOB-level runtime env ``init_ray`` sets — no
    per-actor forwarding needed.
    """
    task_to_env: dict[str, EnvironmentInterface] = {}
    task_to_env_config: dict[str, Any] = {}

    if "cudagym" in env_configs:
        for env_name, cfg in env_configs["cudagym"].items():
            # Every entry declares its own `sku:`; the actor rejects a missing
            # or non-canonical one rather than guessing from the entry name.
            env = CudaGymEnvironment.options(  # type: ignore[attr-defined]
                num_gpus=0,  # HTTP client only; GPU work runs on the CudaGym server
                runtime_env={"py_executable": get_actor_python_env(_CUDAGYM_ENV_FQN)},
            ).remote(cfg)
            task_to_env[env_name] = env
            # Read the typed config back from the actor: validated, with the
            # defaults (weight, sku) the data layer reads already filled.
            task_to_env_config[env_name] = ray.get(env.get_eval_config.remote())
            # Fail fast if the eval endpoint reports different silicon than the
            # entry's sku (no-op when verify_endpoint_sku is false).
            ray.get(env.verify_endpoint_sku.remote())

    if not task_to_env:
        raise ValueError(f"No 'cudagym' environment found in env config: {env_configs}")
    return task_to_env, task_to_env_config


def main() -> None:
    """Main entry point; mirrors the other ``run_grpo_*`` example scripts."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__),
            "configs",
            "recipes",
            "atlas",
            "grpo_cuda_qwen3-8b.yaml",
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
    task_to_env, task_to_env_config = setup_environments(config.env)

    dataset, val_dataset = setup_data(
        tokenizer, config.data, config.grpo, task_to_env_config
    )

    # setup() builds the whole training stack; the NeMo-Gym handle and teacher
    # worker groups are unused in this single-turn recipe.
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
