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

Two rollout modes share one evaluation core:
  * **single-turn** — a native ``EnvironmentInterface``
    (``cudagym_environment.CudaGymEnvironment``). The policy emits one
    ``<think>`` + fenced-kernel completion; the environment evaluates it on a
    CudaGym server and returns the correctness-gated reward. Lives in this
    package.
  * **agentic** — a NeMo-Gym ``cuda_agent`` wrapping OpenCode (see
    ``3rdparty/Gym-workspace/Gym/responses_api_agents/cuda_agent``). The policy
    iterates write -> ``cudagym evaluate`` -> read across turns; its resources
    server (``3rdparty/Gym-workspace/Gym/resources_servers/cudagym/app.py``)
    vendors a copy of this package's evaluation and reward logic.

Both modes build typed CudaGym ``Solution``/``Definition``/``Workload`` objects,
call ``cudagym.sdk.workflows.evaluate`` (see ``cudagym_client``), and score the
returned ``Trace`` with the correctness-gated reward (see ``reward``).

Only the lightweight, dependency-free containers are re-exported here so that
``import nemo_rl.environments.atlas`` does not pull in ``cudagym``/``ray``. The
Ray actor is imported by its fully qualified name
(``nemo_rl.environments.atlas.cudagym_environment.CudaGymEnvironment``, see
``nemo_rl/distributed/ray_actor_environment_registry.py``).
"""

from .cuda_kernel_utils import CudaGymEvalConfig, KernelEvalResult

__all__ = ["CudaGymEvalConfig", "KernelEvalResult"]
