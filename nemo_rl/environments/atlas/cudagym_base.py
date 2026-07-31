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

``BaseCudaEvaluator`` is a mixin: it turns (prompt, completion, problem-metadata)
triples into ``KernelEvalResult``s (parse -> build typed ``Solution`` -> run the
CudaGym SDK -> map the ``Trace``) and scores them with the correctness-gated
reward (``reward.get_reward``). The single-turn env mixes this in; the
agentic path reuses ``cudagym_client`` + ``reward`` directly.

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

from cudagym.sdk import Client
from cudagym.sdk.errors import CudaGymCompilationError, CudaGymExecutionError

from . import cudagym_client, reward
from .cuda_kernel_utils import CudaGymEvalConfig, KernelEvalResult, fence_lang_for
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
        prompts: list[str],
        completions: list[str],
        metadata_list: list[dict],
    ) -> list[KernelEvalResult]:
        """Evaluate (prompt, completion, metadata) triples concurrently.

        Each ``metadata`` dict carries one KernelFactory problem: ``definition``
        (dict), ``workloads`` (list[dict]), ``language``, ``target_hardware``,
        ``destination_passing_style``.

        Returns one ``KernelEvalResult`` per input, in order. All failures are
        captured on the result (never raised) so a single malformed or
        non-compiling completion cannot fail the whole batch.
        """
        assert len(prompts) == len(completions) == len(metadata_list), (
            "evaluate_batch inputs must have equal length"
        )
        results = [
            KernelEvalResult(original_prompt=p, original_completion=c)
            for p, c in zip(prompts, completions)
        ]

        async def _evaluate_one(idx: int) -> None:
            result = results[idx]
            meta = metadata_list[idx]
            language = meta["language"]
            fence_lang = fence_lang_for(language)

            # Stage 1 (format): parse <think>+fenced code -> typed Solution.
            if not check_inline_format(completions[idx], fence_lang):
                result.metadata["format_error"] = (
                    "completion is not <think>...</think> followed by a fenced code block"
                )
                return
            try:
                code = get_code(completions[idx], fence_lang)
                definition, workloads = cudagym_client.parse_problem(meta)
                row_sku = meta.get("target_hardware")
                if row_sku and self.eval_config.sku and row_sku.upper() != self.eval_config.sku.upper():
                    # The endpoint handshake verifies eval_config.sku, but this
                    # is the value that reaches Solution.spec.target_hardware --
                    # so a row declaring B200 against an H100 endpoint compiled
                    # for the wrong silicon and every sample scored 0, reading
                    # as "the model cannot write kernels".
                    result.metadata["config_error"] = (
                        f"row target_hardware={row_sku!r} != env sku={self.eval_config.sku!r}; "
                        "the endpoint is verified against the env sku, so this row would build for other silicon"
                    )
                    return
                solution = cudagym_client.build_solution(
                    code=code,
                    language=language,
                    definition_name=definition.name,
                    target_hardware=row_sku or self.eval_config.sku,
                    destination_passing_style=meta.get(
                        "destination_passing_style", True
                    ),
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
            except Exception as e:  # noqa: BLE001 - surface unexpected SDK/transport errors
                result.metadata["evaluation_error"] = str(e)
                return
            cudagym_client.update_result_from_trace(
                trace, result, sol_anchors=meta.get("sol_anchors")
            )

        await asyncio.gather(*(_evaluate_one(i) for i in range(len(results))))
        return results

    def get_reward(self, result: KernelEvalResult) -> float:
        """Correctness-gated reward for one result (see ``reward.get_reward``)."""
        return reward.get_reward(
            result,
            self.eval_config.reward_weights,
            self.eval_config.perf_reward_config,
        )
