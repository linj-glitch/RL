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

"""Single-turn (M0) CudaGym GRPO environment.

The policy emits ONE completion per prompt (``<think>...</think>`` + a fenced
kernel); this env evaluates it on CudaGym and returns a staged reward, then
terminates (``done = 1`` for every sample). It is the de-risking baseline for
the agentic path (M1) and shares all evaluation + reward logic with it via
``BaseCudaEvaluator`` (``cudagym_base``).

Flow per ``step`` (everything is batched):
  message_log_batch -> (first user prompt, last assistant completion) per sample
    -> ``evaluate_batch`` (parse -> build Solution -> cudagym evaluate -> Trace -> KernelEvalResult)
    -> ``get_reward`` (staged partial credit)
    -> EnvironmentReturn(observations, metadata+{correctness,speedup}, next_stop_strings=None,
                         rewards=Tensor[B], terminateds=ones, answers=None)
GRPO then trains the assistant tokens (``<think>`` + code) against this reward;
prompt/observation tokens are masked by provenance.
"""

import asyncio
from typing import Literal, TypedDict

import numpy as np
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from .cuda_kernel_utils import CudaGymEvalConfig
from .cudagym_base import BaseCudaEvaluator


class CudaGymEnvironmentMetadata(TypedDict, total=False):
    """Per-sample state passed to ``step`` (the datum's ``extra_env_info``).

    Carries the SOLBench/KFB problem so the evaluator can build the typed
    ``Solution``/``Definition``/``Workload`` objects. ``correctness``/``speedup``
    are written back on the way out for ``global_post_process_and_metrics``.
    """

    language: str  # cudagym SupportedLanguages value (e.g. "triton", "cuda_cpp")
    definition: dict  # a Definition dict (e.g. a KFB definition.json)
    workloads: list  # list of Workload dicts (e.g. KFB workload.jsonl lines)
    target_hardware: str  # cudagym SupportedHardware (e.g. "B200"); falls back to env arch
    destination_passing_style: bool
    correctness: bool
    speedup: float


@ray.remote  # pragma: no cover
class CudaGymEnvironment(EnvironmentInterface, BaseCudaEvaluator):
    def __init__(self, config: CudaGymEvalConfig | dict):
        # Accept either a typed config or the raw YAML dict (env.cudagym.<arch>).
        if isinstance(config, CudaGymEvalConfig):
            self.eval_config = config
        else:
            defaults = CudaGymEvalConfig()
            self.eval_config = CudaGymEvalConfig(
                arch=config.get("arch", defaults.arch),
                weight=config.get("weight", defaults.weight),
                compilation_timeout=config.get(
                    "compilation_timeout", defaults.compilation_timeout
                ),
                execution_timeout_per_trial=config.get(
                    "execution_timeout_per_trial", defaults.execution_timeout_per_trial
                ),
                reward_weights=config.get("reward_weights", defaults.reward_weights),
                perf_reward_config=config.get(
                    "perf_reward_config", defaults.perf_reward_config
                ),
                benchmark_config=config.get("benchmark_config", defaults.benchmark_config),
                server_url=config.get("server_url", defaults.server_url),
                auth_token=config.get("auth_token", defaults.auth_token),
            )

        if not self.eval_config.arch:
            raise ValueError("CudaGymEnvironment requires 'arch' (the target GPU, e.g. B200)")

        # One transport client + one event loop per actor. The client is
        # imported here (not at module top) so the data layer can import the
        # config without pulling in cudagym. When ``server_url`` is unset we use
        # CudaGymClient.from_env() (reads CUDAGYM_UNIFIED_SERVER_URL/AUTH_TOKEN),
        # which is how colocated mode injects the Ray-head address.
        from cudagym.sdk import CudaGymClient

        if self.eval_config.server_url:
            self._client = CudaGymClient(
                server_url=self.eval_config.server_url,
                auth_token=self.eval_config.auth_token,
            )
        else:
            self._client = CudaGymClient.from_env()

        # aiohttp sessions bind to the running loop; always drive evaluate_batch
        # on this single owned loop so the session stays consistent across steps.
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def get_eval_config(self) -> CudaGymEvalConfig:
        """Return this env's evaluation config (read by run_grpo_cuda's data setup)."""
        return self.eval_config

    def step(
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[CudaGymEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Evaluate one completion per sample and return a single-turn reward.

        Args:
            message_log_batch: per-sample OpenAI-style message logs. We take the
                first ``user`` message as the prompt and the last ``assistant``
                message as the completion to evaluate.
            metadata: per-sample ``CudaGymEnvironmentMetadata`` (the problem).

        Returns:
            ``EnvironmentReturn`` with ``terminateds`` all True (single-turn).
        """
        user_prompt_batch: list[str] = []
        completion_batch: list[str] = []
        for conversation in message_log_batch:
            user_prompts = [m["content"] for m in conversation if m["role"] == "user"]
            assistant_msgs = [m["content"] for m in conversation if m["role"] == "assistant"]
            user_prompt_batch.append(user_prompts[0] if user_prompts else "")
            completion_batch.append(assistant_msgs[-1] if assistant_msgs else "")

        # Build the typed Solution per sample, compile+execute on CudaGym, and
        # map each Trace -> KernelEvalResult (all concurrent, on the owned loop).
        results = self._loop.run_until_complete(
            self.evaluate_batch(user_prompt_batch, completion_batch, metadata)
        )
        rewards = [self.get_reward(result) for result in results]

        # Observation = human-readable eval feedback. Unused after a single turn
        # (the episode terminates) but kept for parity with the agentic path and
        # for debugging printed rollouts.
        observations = [
            {
                "role": "environment",
                "content": (
                    f"Reward: {rew:.3f}\n"
                    f"Compiled: {result.compiled}\n"
                    f"Executed: {result.executed}\n"
                    f"Correct: {result.correctness}\n"
                    f"Speedup: {result.speedup:.3f}x\n"
                    f"Metadata: {result.metadata}\n"
                ),
            }
            for rew, result in zip(rewards, results)
        ]

        # Carry correctness/speedup forward for global metrics aggregation.
        output_metadata: list[CudaGymEnvironmentMetadata] = []
        for meta, result in zip(metadata, results):
            updated = dict(meta)
            updated["correctness"] = bool(result.correctness)
            updated["speedup"] = float(result.speedup)
            output_metadata.append(updated)  # type: ignore[arg-type]

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).cpu()
        terminateds = torch.ones_like(rewards_tensor).cpu()  # single-turn: always done
        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=output_metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds,
            answers=None,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        """Aggregate correctness rate + mean speedup over correct kernels."""
        metrics: dict = {}
        try:
            batch_metadata = batch.get("metadata", [])
            if batch_metadata:
                correctness = np.array(
                    [bool(m.get("correctness", False)) for m in batch_metadata]
                )
                speedups = np.array(
                    [float(m.get("speedup", -1.0)) for m in batch_metadata]
                )
                metrics["correctness_rate"] = (
                    float(correctness.mean()) if len(correctness) else 0.0
                )
                # Average speedup only over kernels that were actually correct.
                metrics["avg_speedup"] = (
                    float(speedups[correctness].mean()) if correctness.any() else 0.0
                )
        except Exception:
            pass
        return batch, metrics

    def shutdown(self) -> None:
        """Close the transport session + event loop."""
        try:
            self._loop.run_until_complete(self._client.close())
        finally:
            self._loop.close()
