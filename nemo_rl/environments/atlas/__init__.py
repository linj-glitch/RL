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

"""Atlas: GRPO on CudaGym / KernelFactory kernel-optimization problems.

Two rollout shapes share one core:
  * **single-turn** — a native ``EnvironmentInterface``
    (``cudagym_environment.CudaGymEnvironment``): the policy emits one
    ``<think>`` + fenced-kernel completion, the env evaluates it on CudaGym and
    returns a staged reward. Lives in this package.
  * **agentic** — a NeMo-Gym ``cuda_agent`` wrapping OpenCode (see
    ``3rdparty/Gym-workspace/Gym/responses_api_agents/cuda_agent``); the policy
    iterates write -> ``cudagym evaluate`` -> read across turns.

Both build typed CudaGym ``Solution``/``Definition``/``Workload`` objects and
call ``cudagym.sdk.workflows.evaluate`` (``cudagym_client``), then score the
returned ``Trace`` with the staged reward (``reward``).

Only the lightweight, dependency-free containers are re-exported here so that
``import nemo_rl.environments.atlas`` does not pull in ``cudagym``/``ray``; the
Ray actor is imported by FQN (``...atlas.cudagym_environment.CudaGymEnvironment``).
"""

from .cuda_kernel_utils import CudaGymEvalConfig, KernelEvalResult

__all__ = ["CudaGymEvalConfig", "KernelEvalResult"]
