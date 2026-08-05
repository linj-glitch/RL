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

"""Single-turn CudaGym GRPO environment.

The policy emits ONE completion per prompt (``<think>...</think>`` + a fenced
kernel); this env evaluates it on CudaGym and returns the correctness-gated
reward, then terminates (``done = 1`` for every sample). It is the simpler
counterpart of the agentic path (see the package docstring in ``__init__.py``)
and shares the evaluation + reward logic with it via ``BaseCudaEvaluator``
(``cudagym_base``).

Flow per ``step`` (everything is batched):

  1. Split each sample's message log into the first ``user`` prompt and the
     last ``assistant`` completion.
  2. ``evaluate_batch`` parses each completion, builds the typed ``Solution``,
     runs the CudaGym evaluation, and maps the ``Trace`` to a ``KernelEvalResult``.
  3. ``get_reward`` scores each result with the correctness-gated reward.
  4. The ``EnvironmentReturn`` carries the feedback observations, the metadata
     with {correctness, speedup, human_best_speedup, sol_score} written back,
     ``rewards`` as a Tensor[B], ``terminateds`` all ones, and no stop strings.

GRPO then trains the assistant tokens (``<think>`` + code) against this reward;
prompt and environment-observation tokens are excluded from the loss.
"""

import logging
import os
from typing import TypedDict

import ray
import torch
from cudagym.sdk import Client
from pydantic import ValidationError

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

from . import cudagym_client
from .cuda_kernel_utils import (
    CudaGymEvalConfig,
    aggregate_kernel_metrics,
    canonical_sku,
    verify_health_payload,
)
from .cudagym_base import BaseCudaEvaluator

LOG = logging.getLogger(__name__)


class CudaGymEnvironmentMetadata(TypedDict, total=False):
    """Per-sample state passed to ``step`` (the datum's ``extra_env_info``).

    Carries the KernelFactory problem so the evaluator can build the typed
    ``Solution``/``Definition``/``Workload`` objects. ``correctness`` and the
    three performance signals (``speedup``, ``human_best_speedup``,
    ``sol_score``) are written back on the way out for
    ``global_post_process_and_metrics``.
    """

    language: str  # cudagym SupportedLanguages value (e.g. "triton", "cuda_cpp")
    definition: dict  # a Definition dict (e.g. a KFB definition.json)
    workloads: list  # list of Workload dicts (e.g. KFB workload.jsonl lines)
    target_hardware: (
        str  # cudagym SupportedHardware (e.g. "B200"); falls back to env sku
    )
    destination_passing_style: bool
    sol_anchors: dict  # workload uuid -> {human_best_latency_ms, sol_latency_ms}
    correctness: bool
    speedup: float  # speedup over the eager PyTorch reference (cudagym speedup_factor)
    human_best_speedup: float  # speedup over the human-best / optimized-baseline anchor
    sol_score: float  # anchored SOL score in [0,1] (== the performance reward term)


@ray.remote  # pragma: no cover
class CudaGymEnvironment(EnvironmentInterface, BaseCudaEvaluator):
    """Ray actor for single-turn kernel evaluation: a thin CudaGym HTTP client (no GPUs)."""

    def __init__(self, config: dict):
        """Validate the raw ``env.cudagym.<name>`` mapping and build the SDK client."""
        # The raw YAML dict (env.cudagym.<sku>) is the only calling convention:
        # the driver passes the recipe mapping straight through Ray. The model
        # forbids extra keys (the same class of typo that
        # validate_benchmark_config catches one level down), and pydantic's own
        # error names only the offending keys, so the valid ones are added here.
        try:
            self.eval_config = CudaGymEvalConfig(**config)
        except ValidationError as exc:
            raise ValueError(
                f"invalid env.cudagym entry: {exc}\n"
                f"valid keys: {sorted(CudaGymEvalConfig.model_fields)}"
            ) from exc

        if not self.eval_config.sku:
            raise ValueError(
                "CudaGymEnvironment requires 'sku' (the target GPU, e.g. B200)"
            )
        # Fail on a typo'd benchmark_config key here rather than letting pydantic
        # drop it: a silently ignored `lock_clocks` leaves clocks unlocked while
        # timings are scored against locked-clock anchors.
        cudagym_client.validate_benchmark_config(self.eval_config.benchmark_config)
        # And on a sku the SDK's Solution schema would refuse (see canonical_sku:
        # the enum is case-sensitive, so a near-miss survives every check up to
        # build_solution and is then blamed on the model).
        canonical_sku(self.eval_config.sku, "env.cudagym.<name>.sku")

        # One transport client + one event loop per actor. cudagym >= 2.x
        # speaks split compile/GPU URLs; our launch plumbing carries ONE
        # unified endpoint (the in-allocation LB or a managed remote), so it
        # is passed as both. Resolution order: the recipe-pinned server_url,
        # then CUDAGYM_UNIFIED_SERVER_URL / CUDAGYM_URL (how colocated mode
        # injects the LB address), then Client.from_env() (native split
        # CUDAGYM_{COMPILE,GPU}_SERVER_URL; raises with a clear message when
        # nothing is set).
        server_url = (
            self.eval_config.server_url
            or os.environ.get("CUDAGYM_UNIFIED_SERVER_URL")
            or os.environ.get("CUDAGYM_URL")
        )
        if server_url:
            self._client = Client(
                compile_server_url=server_url,
                gpu_server_url=server_url,
                # A pinned server_url should still honor the ambient token —
                # e.g. remote/Modal endpoints with a recipe-pinned URL.
                auth_token=self.eval_config.auth_token
                or os.environ.get("CUDAGYM_AUTH_TOKEN"),
            )
        else:
            self._client = Client.from_env()

        # NOTE: evaluate_batch (BaseCudaEvaluator) is async, which makes Ray run
        # this actor in asyncio mode: every method executes on the actor's own
        # event loop, so async methods just `await` (aiohttp sessions stay bound
        # to that one loop). Never call loop.run_until_complete in here — the
        # loop is already running.

    async def verify_endpoint_sku(self) -> None:
        """Fail fast when the eval endpoint's silicon doesn't match ``sku``.

        A mismatch is otherwise SILENT for Triton kernels (they JIT-compile on
        whatever GPU serves the request and return that GPU's timings). The
        driver (``examples/run_grpo_cuda.py``) calls this right after actor
        creation — after ``ray.sub``'s server health checks, so the endpoint is
        already up. No-op when ``verify_endpoint_sku`` is disabled in the config.
        """
        if not self.eval_config.verify_endpoint_sku:
            return
        resp = await self._client.health()
        # A load-balanced endpoint nests per-server payloads under "servers";
        # a single server reports one "health" dict at top level.
        payloads = (
            [s.get("health") or {} for s in resp["servers"]]
            if "servers" in resp
            else [resp.get("health") or {}]
        )
        if resp.get("status") != "healthy":
            raise ValueError(
                f"CudaGym endpoint unhealthy at env init (sku={self.eval_config.sku}): {resp}"
            )
        # Check EVERY pool member (like the Gym resources server's copy): one
        # matching server must not vouch for a heterogeneous pool. The verdict
        # is three-valued — False is a confirmed mismatch (raise), None means
        # the payload reports nothing checkable (warn; compile-only responders
        # have no GPU fields), True is a confirmed match.
        confirmed = False
        for payload in payloads:
            ok, detail = verify_health_payload(payload, self.eval_config.sku or "")
            if ok is False:
                raise ValueError(
                    f"CudaGym endpoint SKU mismatch for env sku={self.eval_config.sku}: {detail} "
                    f"(set verify_endpoint_sku: false to override deliberately)"
                )
            if ok is None:
                LOG.warning(
                    "cudagym endpoint SKU check (%s): %s", self.eval_config.sku, detail
                )
            else:
                confirmed = True
                LOG.info(
                    "cudagym endpoint SKU check (%s): %s", self.eval_config.sku, detail
                )
        # An endpoint where nothing could be checked is reported, not passed:
        # silent non-verification is how a silicon mismatch hides.
        if not confirmed:
            LOG.warning(
                "cudagym endpoint SKU NOT VERIFIED (sku=%s): no pool member reported a checkable GPU",
                self.eval_config.sku,
            )

    def get_eval_config(self) -> CudaGymEvalConfig:
        """Return this env's evaluation config (read by ``examples/run_grpo_cuda.py`` during data setup)."""
        return self.eval_config

    async def step(
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
        # Split each conversation into the prompt and the completion to evaluate.
        user_prompt_batch: list[str] = []
        completion_batch: list[str] = []
        for conversation in message_log_batch:
            user_prompts = [m["content"] for m in conversation if m["role"] == "user"]
            assistant_msgs = [
                m["content"] for m in conversation if m["role"] == "assistant"
            ]
            user_prompt_batch.append(user_prompts[0] if user_prompts else "")
            completion_batch.append(assistant_msgs[-1] if assistant_msgs else "")

        # Build the typed Solution per sample, compile+execute on CudaGym, and
        # map each Trace -> KernelEvalResult (all concurrent on the actor loop).
        results = await self.evaluate_batch(
            user_prompt_batch, completion_batch, metadata
        )
        rewards = [self.get_reward(result) for result in results]

        # Observation = human-readable eval feedback (episode terminates after it).
        # Clip hard: the rollout loop appends+tokenizes the observation even on
        # terminated episodes, so an unclipped multi-KB log pads the sequence and
        # falsely flips `truncated` (and with grpo.overlong_filtering would zero
        # failed-kernel gradients).
        def _clip(text: object, limit: int = 2000) -> str:
            s = str(text)
            return (
                s
                if len(s) <= limit
                else s[:limit] + f"...[clipped {len(s) - limit} chars]"
            )

        observations = [
            {
                "role": "environment",
                "content": (
                    f"Reward: {rew:.3f}\n"
                    f"Compiled: {result.compiled}\n"
                    f"Executed: {result.executed}\n"
                    f"Correct: {result.correctness}\n"
                    f"Speedup: {result.speedup:.3f}x\n"
                    f"Metadata: {_clip(result.metadata)}\n"
                ),
            }
            for rew, result in zip(rewards, results)
        ]

        # Carry correctness + the three perf signals forward for global metrics.
        output_metadata: list[CudaGymEnvironmentMetadata] = []
        for meta, result in zip(metadata, results):
            updated = dict(meta)
            updated["correctness"] = bool(result.correctness)
            updated["speedup"] = float(result.speedup)  # vs eager reference
            updated["human_best_speedup"] = float(
                result.human_best_speedup
            )  # vs human-best/baseline
            updated["sol_score"] = float(result.sol_score)  # anchored SOL score
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
        """Aggregate the step's reward-observability metrics.

        Delegates to ``cuda_kernel_utils.aggregate_kernel_metrics`` (shared with the
        NeMo-Gym agentic path so both log identical names/semantics): ``correctness_rate``
        over the batch; then, over the CORRECT kernels only, ``avg_speedup_over_ref``
        (vs the eager PyTorch reference), ``avg_speedup_over_baseline`` (vs the
        human-best baseline anchor), ``avg_sol_score`` (the anchored SOL score that IS
        the performance reward), and ``perf_ref_fallback_rate`` (fraction whose perf
        reward fell back to speedup-over-ref for lack of a SOL/human-best anchor).
        """
        metrics: dict = {}
        try:
            batch_metadata = batch.get("metadata", []) or []
            records = [m for m in batch_metadata if isinstance(m, dict)]
            metrics = aggregate_kernel_metrics(records)
        except Exception as e:  # never let metric aggregation crash a rollout
            LOG.warning("Error aggregating cudagym metrics: %r", e)
        return batch, metrics

    async def shutdown(self) -> None:
        """Close the transport session."""
        await self._client.close()
