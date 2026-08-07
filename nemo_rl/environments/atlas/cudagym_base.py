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

"""Shared CudaGym evaluation + reward for the atlas environments.

``BaseCudaEvaluator`` is a mixin: it turns (completion, problem-metadata)
pairs into ``KernelEvalResult``s (it parses each completion, builds the typed
``Solution``, runs the CudaGym SDK, and maps the ``Trace``) and scores them with
the correctness-gated reward (``reward.get_reward``). The single-turn env mixes
this in. The agentic path's resources server
(``3rdparty/Gym-workspace/Gym/resources_servers/cudagym/app.py``) drives the
same SDK and the same ``cudagym.rl`` scoring helpers through its own glue.

Evaluation is async because the SDK client is aiohttp-based; the owning Ray
actor supplies a live ``Client`` as ``self._client`` and drives
``evaluate_batch`` on its own event loop. Hard build/run failures raise
``CudaGym{Compilation,Execution}Error`` (caught here so one bad completion can't
fail the batch); per-workload correctness/runtime failures come back in the
``Trace`` and are mapped by ``cudagym_client.update_result_from_trace``.
"""

from __future__ import annotations

import asyncio
from abc import ABC
from collections.abc import Mapping, Sequence
from typing import Any

from cudagym.rl import build_solution, fence_lang_for
from cudagym.sdk import Client
from cudagym.sdk.errors import CudaGymCompilationError, CudaGymExecutionError

from . import cudagym_client, reward
from .cuda_kernel_utils import CudaGymEvalConfig, KernelEvalResult
from .llm_response_parsing import check_inline_format, get_code


class BaseCudaEvaluator(ABC):
    """Evaluate + reward a batch of kernel completions against CudaGym.

    Subclasses (the Ray actors) must set ``self.eval_config`` (a
    ``CudaGymEvalConfig``) and ``self._client`` (a ``Client``).
    """

    eval_config: CudaGymEvalConfig
    _client: Client

    async def evaluate_batch(
        self,
        completions: list[str],
        metadata_list: Sequence[Mapping[str, Any]],
    ) -> list[KernelEvalResult]:
        """Evaluate (completion, metadata) pairs concurrently.

        Each ``metadata`` dict carries one KernelFactory problem: ``definition``
        (dict), ``workloads`` (list[dict]), ``language``, ``target_hardware``,
        ``destination_passing_style``.

        Returns one ``KernelEvalResult`` per input, in order. Completion-level
        failures (bad format, failed build, evaluation errors) are captured on
        the result rather than raised, so a single malformed or non-compiling
        completion cannot fail the whole batch. Malformed problem *metadata*
        (a missing or unknown ``language``, a missing
        ``destination_passing_style``) does raise: that is a dataset bug, not a
        model output, and it should stop the run.
        """
        assert len(completions) == len(metadata_list), (
            "evaluate_batch inputs must have equal length"
        )
        results = [KernelEvalResult() for _ in completions]

        async def _evaluate_one(idx: int) -> None:
            """Evaluate one (completion, problem) pair onto ``results[idx]``."""
            result = results[idx]
            meta = metadata_list[idx]
            # Hard-indexed like language: the data processor always writes the
            # key, so a missing one is a dataset bug and should stop the run,
            # not read as the model's format error.
            language = meta["language"]
            destination_passing_style = meta["destination_passing_style"]
            fence_lang = fence_lang_for(language)

            # Stage 1 (format): parse <think>+fenced code -> typed Solution.
            if not check_inline_format(completions[idx], fence_lang):
                result.metadata["format_error"] = (
                    "completion is not <think>...</think> followed by a fenced code block"
                )
                return
            # Row validation first: definition/workloads are dataset data, so a
            # malformed row is a config error, not the model's format error.
            try:
                definition, workloads = cudagym_client.parse_problem(meta)
            except Exception as e:  # noqa: BLE001 - record the row problem verbatim
                result.metadata["config_error"] = f"invalid problem row: {e}"
                return
            # Rest of stage 1: extract the fenced code and build the typed Solution.
            try:
                code = get_code(completions[idx], fence_lang)
                row_sku = meta.get("target_hardware")
                if (
                    row_sku
                    and self.eval_config.sku
                    and row_sku.upper() != self.eval_config.sku.upper()
                ):
                    # This value becomes Solution.spec.target_hardware, but only
                    # eval_config.sku is endpoint-verified — a mismatched row
                    # would silently build for the wrong silicon, so refuse it.
                    result.metadata["config_error"] = (
                        f"row target_hardware={row_sku!r} != env sku={self.eval_config.sku!r}; "
                        "the endpoint is verified against the env sku, so this row would build for other silicon"
                    )
                    return
                target_hardware = row_sku or self.eval_config.sku
                if not target_hardware:
                    # The GPU to build for comes from the row or the env
                    # config; a missing one is a config bug, not a model error.
                    result.metadata["config_error"] = (
                        "the row pins no target_hardware and the env config declares no sku"
                    )
                    return
                solution = build_solution(
                    code=code,
                    language=language,
                    definition_name=definition.name,
                    target_hardware=target_hardware,
                    destination_passing_style=destination_passing_style,
                    author="nemorl",
                )
            except Exception as e:  # noqa: BLE001 - any parse/validation error is a format error
                result.metadata["format_error"] = f"failed to build solution: {e}"
                return
            result.formatted = True

            # Stage 2 (compile + execute + parse): run the SDK, map Trace -> result.
            # A C++ build failure raises CudaGymCompilationError (compiled stays
            # False); a GPU-job crash raises CudaGymExecutionError (it compiled,
            # but couldn't run); otherwise per-workload outcomes are in the Trace.
            try:
                trace = await cudagym_client.evaluate_solution(
                    self._client, solution, definition, workloads, self.eval_config
                )
            except CudaGymCompilationError as e:
                result.metadata["compile_error"] = str(e)
                return
            except CudaGymExecutionError as e:
                result.compiled = True
                result.metadata["execution_error"] = str(e)
                return
            except Exception as e:  # noqa: BLE001 - record unexpected SDK/transport errors
                result.metadata["evaluation_error"] = str(e)
                return
            # Mapping the Trace onto the result is per-sample work like the
            # stages above: an unexpected exception here must mark this sample
            # as an evaluation error, not fail the whole batch.
            try:
                cudagym_client.update_result_from_trace(
                    trace, result, sol_anchors=meta.get("sol_anchors")
                )
            except Exception as e:  # noqa: BLE001 - record unexpected mapping errors
                result.metadata["evaluation_error"] = f"error mapping trace: {e}"

        await asyncio.gather(*(_evaluate_one(i) for i in range(len(results))))
        return results

    def get_reward(self, result: KernelEvalResult) -> float:
        """Correctness-gated reward for one result (see ``reward.get_reward``)."""
        return reward.get_reward(
            result,
            self.eval_config.reward_weights,
            self.eval_config.perf_reward_config,
        )
