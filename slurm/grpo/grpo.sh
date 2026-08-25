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

# ========================= user defined variables =========================
export EXP_NAME=${EXP_NAME:-DEFAULT_EXP_NAME}
export CONFIG_NAME=${CONFIG_NAME:-DEFAULT_CONFIG_NAME}
export EXTRA_CONFIG_OPTS=${EXTRA_CONFIG_OPTS:-DEFAULT_EXTRA_CONFIG_OPTS}

# Required paths. Use absolute paths to avoid mounting the slow /home directory.
export WORKSPACE_PATH=${WORKSPACE_PATH:-DEFAULT_WORKSPACE}
export CACHE_PATH=${CACHE_PATH:-DEFAULT_CACHE}
export CONTAINER=${CONTAINER:-DEFAULT_CONTAINER}
export MODELS_PATH=${MODELS_PATH:-DEFAULT_MODELS}
export OUTPUT_ROOT=${OUTPUT_ROOT:-DEFAULT_OUTPUT}
export DATASETS_PATH=${DATASETS_PATH:-DEFAULT_DATASETS}

# Optional settings for CudaGym (colocated/disjoint in-allocation hosting; the
# `endpoint` kind launches no servers and only needs the URL below).
export CUDAGYM_MODE=${CUDAGYM_MODE:-DEFAULT_CUDAGYM_MODE}
export CUDAGYM_ENABLED=${CUDAGYM_ENABLED:-DEFAULT_CUDAGYM_ENABLED}
export CUDAGYM_CONTAINER=${CUDAGYM_CONTAINER:-DEFAULT_CUDAGYM_CONTAINER}
export ARTIFACTS_DIR=${ARTIFACTS_DIR:-DEFAULT_ARTIFACTS_DIR}
export CCACHE_DIR=${CCACHE_DIR:-DEFAULT_CCACHE_DIR}
export CUDAGYM_NUM_NODES=${CUDAGYM_NUM_NODES:-DEFAULT_CUDAGYM_NUM_NODES}
# endpoint kind: the resolved eval endpoint URL (submit_grpo fills it from the
# recipe's hosting block / registry). Colocated/disjoint runs ignore this —
# ray.sub exports the in-allocation LB URL instead.
export CUDAGYM_UNIFIED_SERVER_URL=${CUDAGYM_UNIFIED_SERVER_URL:-DEFAULT_CUDAGYM_UNIFIED_SERVER_URL}

# API keys
export WANDB_PROJECT="${WANDB_PROJECT:-atlas_nemorl}"
export WANDB_API_KEY=${WANDB_API_KEY:-DEFAULT_WANDB_API_KEY}
export HF_TOKEN=${HF_TOKEN:-DEFAULT_HF_TOKEN}
export CUDAGYM_AUTH_TOKEN=${CUDAGYM_AUTH_TOKEN:-DEFAULT_CUDAGYM_AUTH_TOKEN}
# Modal edge proxy-auth for .modal.run eval endpoints (the `endpoint` hosting kind).
export MODAL_PROXY_TOKEN_ID=${MODAL_PROXY_TOKEN_ID:-DEFAULT_MODAL_PROXY_TOKEN_ID}
export MODAL_PROXY_TOKEN_SECRET=${MODAL_PROXY_TOKEN_SECRET:-DEFAULT_MODAL_PROXY_TOKEN_SECRET}

# Slurm allocation options
export NUM_NODES=${NUM_NODES:-DEFAULT_NUM_NODES}
export TIME=${TIME:-DEFAULT_TIME}
export GPUS_PER_NODE=${GPUS_PER_NODE:-DEFAULT_GPUS_PER_NODE}
# =======================================================================

# uv wheels: NODE-LOCAL cache seeded from one Lustre tarball. Both simpler
# variants lose: no seed = ~13 min of downloads per node (27b-9); a
# Lustre-MOUNTED live cache = copy-mode installs + flock contention across the
# ~10 concurrent per-actor venv builds (27b-12: policy init 28s -> 35 min).
# SETUP_COMMAND below extracts the tar into each container's /root/.cache
# (merging with the image's own cache — NEVER mount over /root/.cache/uv, the
# image venv symlinks into it), so uv installs hardlink at local-disk speed
# with zero network. Refresh the seed after a lock bump:
#   cd $CACHE_PATH && tar -cf uv-cache-seed.tar.tmp uv && mv uv-cache-seed.tar.tmp uv-cache-seed.tar
export HF_HOME=${CACHE_PATH}/huggingface
# Everything is prewarmed into HF_HOME; offline mode makes hub resolution
# purely cache-deterministic. With it unset, every rank's snapshot_download
# does a network etag check and one hub flake across hundreds of ranks
# poisons path resolution (dsv4-27: "No .safetensors files or index found").
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
# Refit transfer tuning (speedup item 5): 1.6GB default buckets underfill the
# ring; checkpoint-engine-class systems use ~8GiB. 3 buffers overlap
# gather/pack/broadcast.
export NRL_REFIT_BUFFER_MEMORY_RATIO=${NRL_REFIT_BUFFER_MEMORY_RATIO:-0.06}
export NRL_REFIT_NUM_BUFFERS=${NRL_REFIT_NUM_BUFFERS:-3}
export UV_CACHE_SEED_TAR=${UV_CACHE_SEED_TAR:-${CACHE_PATH}/uv-cache-seed.tar}
# vLLM torch.compile/AOT artifacts (NeMo-RL appends a per-engine seed suffix to
# this base) plus the triton/inductor JIT caches: persisting them skips the
# ~1 min first-request compile per engine per job and de-risks first-prefill
# JIT stalls. Written once per config, read thereafter.
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-${CACHE_PATH}/vllm}
# DeepGEMM stays ON: its dsv4-7 ZeroDivision was a symptom of TP32 sharding a
# grouped-BMM weight to zero groups (fixed by TP16), while the cutlass c3x
# fallback rejects V4's block-fp8 shapes at ANY TP (dsv4-8 at TP32, dsv4-12 at
# TP16: dispatch_scaled_mm hpp:17). DeepGEMM is the kernel this model was
# built for.
export VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM:-1}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${CACHE_PATH}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${CACHE_PATH}/inductor}
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true
export OUTPUT_DIR=${OUTPUT_ROOT}/${EXP_NAME}

# Set for clusters whose sbatch rejects --gpus-per-node (applied at the sbatch call).
export SKIP_GRES_ARG=${SKIP_GRES_ARG:-DEFAULT_SKIP_GRES_ARG}

# In disjoint mode the trailing CUDAGYM_NUM_NODES nodes host CudaGym and never
# join Ray — the training config must only request the nodes Ray actually has,
# or virtual-cluster placement fails after full bringup.
if [ "${CUDAGYM_MODE}" = "disjoint" ]; then
    export TRAIN_NUM_NODES=$(( NUM_NODES - ${CUDAGYM_NUM_NODES:-0} ))
else
    export TRAIN_NUM_NODES=${NUM_NODES}
fi

# Runner + uv extras are recipe-dependent (submit_grpo.py fills them):
#   single-turn -> examples/run_grpo_cuda.py           + "--extra atlas"
#   agentic     -> examples/nemo_gym/run_grpo_nemo_gym.py + "--extra atlas --extra nemo_gym"
# --extra atlas pulls the cudagym SDK (thin client) into the venv; the driver and
# the SYSTEM-venv env actors import cudagym.sdk/cudagym.contracts.
export RUN_SCRIPT=${RUN_SCRIPT:-DEFAULT_RUN_SCRIPT}
export UV_EXTRAS=${UV_EXTRAS:-DEFAULT_UV_EXTRAS}

# Sync the venv on every node BEFORE Ray starts, so `ray start` and the
# driver's `uv run` agree on the Ray version (ray.init refuses a skew).
# The interpreter bootstrap runs first when needed: the repo requires the
# Python patch release pinned in .python-version, and an older training image
# bakes only an older interpreter (and a uv whose manifest predates the
# release). `uv python find` honors requires-python, so on a current image the
# bootstrap short-circuits; drop it once no image in use predates the pin.
# The seed extraction runs first (idempotent via the marker file — the head
# retry loop re-runs SETUP_COMMAND): it merges the shared wheel tarball into
# this container's node-local uv cache so the sync below downloads nothing.
export SETUP_COMMAND=${SETUP_COMMAND:-"( [ -f '${UV_CACHE_SEED_TAR}' ] && [ ! -e /root/.cache/.uv_seeded ] && mkdir -p /root/.cache && tar -xf '${UV_CACHE_SEED_TAR}' -C /root/.cache && touch /root/.cache/.uv_seeded && echo '[uv] node-local cache seeded from ${UV_CACHE_SEED_TAR}' || true ) && (uv python find >/dev/null 2>&1 || (curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh && uv python install)) && uv sync ${UV_EXTRAS}"}

# Rebuild the image-baked per-worker venvs (/opt/ray_venvs/*): after a
# dependency bump a stale venv dies unpickling Ray internals.
export NRL_FORCE_REBUILD_VENVS=${NRL_FORCE_REBUILD_VENVS:-true}

# Page-cache prewarm: ray.sub background-reads this directory on every TRAIN
# node while containers extract and venvs sync, so the trainer's
# from_pretrained streams from warm page cache instead of cold Lustre
# (measured 6:57 -> ~1 min for the 27B checkpoint). Derived from the recipe's
# policy model_name; an unset or missing dir skips the warm quietly.
_POLICY_MODEL=$(grep -m1 -E '^[[:space:]]*model_name:' "examples/configs/recipes/atlas/${CONFIG_NAME}" 2>/dev/null | awk '{print $2}' | tr -d '"' | tr -d "'")
if [ -n "$_POLICY_MODEL" ]; then
    export NRL_HF_PREWARM_DIR="${HF_HOME}/hub/models--${_POLICY_MODEL//\//--}"
fi

export COMMAND="uv run ${UV_EXTRAS} ${RUN_SCRIPT} \
    --config examples/configs/recipes/atlas/${CONFIG_NAME} \
    checkpointing.checkpoint_dir=${OUTPUT_DIR}/ckpts \
    logger.log_dir=${OUTPUT_DIR}/logs/wandb \
    logger.wandb.project=${WANDB_PROJECT} \
    logger.wandb.name=${EXP_NAME} \
    logger.wandb_enabled=true \
    cluster.num_nodes=${TRAIN_NUM_NODES} \
    cluster.gpus_per_node=${GPUS_PER_NODE} \
    ${EXTRA_CONFIG_OPTS}
"

# This script runs from inside the uploaded code tree (`cd code && bash ../run.sh`),
# so cwd is the code and its parent is the experiment output dir (logs, ckpts).
cwd=$(pwd -P)
cwd_parent=$(dirname $cwd)

# Mount the experiment dir at its own path, the code tree also at /opt/nemo-rl,
# plus the workspace, model, and dataset roots.
export MOUNTS="$cwd_parent:$cwd_parent,$cwd:/opt/nemo-rl,$WORKSPACE_PATH:$WORKSPACE_PATH,$MODELS_PATH:/models,$DATASETS_PATH:/datasets"
# Import the vendored cudagym SDK straight from the checkout on every node.
# ${PYTHONPATH:+:$PYTHONPATH} appends the previous value only when one is set;
# a plain ":${PYTHONPATH}" would leave a trailing colon when it is unset, which
# Python reads as "also search the process's current directory".
export PYTHONPATH="$cwd/3rdparty/solswarm/cudagym/src${PYTHONPATH:+:$PYTHONPATH}"
# The uploaded cudagym tree has no .git; submit_grpo.py derives this version
# from the submodule at submit time so it tracks bumps.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDAGYM=DEFAULT_CUDAGYM_VERSION
# The SolSwarm checkout: the stitched problem extras + reference-sandbox cache
# key (sandbox_profile: solswarm); the /submit skill source (profile: minimal).
export SOLSWARM_SURFACE_ROOT="$cwd/3rdparty/solswarm"
# Where the agent writes the job's sandbox diagnostics (the pre-flight manifest
# and the once-per-job sandbox_tree.json file-tree snapshot/diff) — on shared
# storage, next to the logs.
export CUDA_AGENT_MANIFEST_DIR="${OUTPUT_DIR}/sandbox_manifests"
# Per-rollout agent containers (container-mode recipes; submit_grpo.py requires
# the flag for them). Empty skips the enroot plumbing; a path bind-mounts the
# host's enroot into the training container, and the Gym agent server reads
# the image path from CUDA_AGENT_ENROOT_IMAGE.
export CUDA_AGENT_ENROOT_IMAGE=${CUDA_AGENT_ENROOT_IMAGE:-DEFAULT_CUDA_AGENT_ENROOT_IMAGE}
if [ -n "$CUDA_AGENT_ENROOT_IMAGE" ]; then
    MOUNTS="$MOUNTS,/usr/lib/enroot:/usr/lib/enroot,/usr/share/enroot:/usr/share/enroot,/etc/enroot:/etc/enroot"
    # enroot's runtime exec's sibling helper binaries by bare name
    # (enroot-nsenter, enroot-mount, enroot-switchroot, ...), so every
    # /usr/bin/enroot* file must ride along, not just the launcher.
    _enroot_bins_found=0
    for _enroot_bin in /usr/bin/enroot*; do
        if [ -e "$_enroot_bin" ]; then
            MOUNTS="$MOUNTS,$_enroot_bin:$_enroot_bin"
            _enroot_bins_found=1
        fi
    done
    # An unmatched glob would silently bind nothing and the agent server would
    # only fail at startup, minutes into the allocation — refuse to submit.
    if [ "$_enroot_bins_found" -eq 0 ]; then
        echo "ERROR: container mode is requested (CUDA_AGENT_ENROOT_IMAGE is set) but /usr/bin/enroot* matches no file on this node, so no enroot binaries can be bind-mounted into the training container. Install enroot or submit without --enroot-agent-image." >&2
        exit 1
    fi
    export MOUNTS
    export SETUP_COMMAND="${SETUP_COMMAND} && (command -v gawk >/dev/null && command -v unsquashfs >/dev/null || (apt-get update -qq && apt-get install -y -qq gawk squashfs-tools))"
    # Node-local scratch for the extracted image: a rootfs is hundreds of
    # thousands of small files, which Lustre handles badly.
    export CUDA_AGENT_ENROOT_DATA=${CUDA_AGENT_ENROOT_DATA:-/tmp/cuda-agent-enroot}
fi

# if -i flag is provided, run the command interactively
if [ "$1" == "-i" ]; then
    unset COMMAND
    echo "Launching interactively..."
fi

export BASE_LOG_DIR=${OUTPUT_DIR}/logs # outside container path
echo ======================================================
echo EXP_NAME: $EXP_NAME
echo CONFIG_NAME: $CONFIG_NAME
echo EXTRA_CONFIG_OPTS: $EXTRA_CONFIG_OPTS
echo WORKSPACE_PATH: $WORKSPACE_PATH
echo CACHE_PATH: $CACHE_PATH
echo CONTAINER: $CONTAINER
echo MODELS_PATH: $MODELS_PATH
echo OUTPUT_DIR: $OUTPUT_DIR
echo DATASETS_PATH: $DATASETS_PATH
echo NUM_NODES: $NUM_NODES
echo TIME: $TIME
echo ======================================================

# Account/partition/qos come from the cluster yaml (submit_grpo.py fills them).
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-DEFAULT_SLURM_ACCOUNT}
export SLURM_PARTITION=${SLURM_PARTITION:-DEFAULT_SLURM_PARTITION}
export SLURM_QOS=${SLURM_QOS:-DEFAULT_SLURM_QOS}

# Assemble the sbatch arguments; ray.sub is the batch script. --dependency=singleton
# queues repeat submissions of the same experiment (--num-jobs) back to back.
SBATCH_ARGS=(
    --nodes=${NUM_NODES} \
    --account=${SLURM_ACCOUNT} \
    --job-name=${SLURM_ACCOUNT}-atlas.grpo.${EXP_NAME} \
    --partition=${SLURM_PARTITION} \
    --dependency=singleton \
    --time=${TIME} \
    --output=${BASE_LOG_DIR}/slurm-%j.out \
)
# QoS-scheduled clusters reject jobs submitted without --qos; empty = no flag.
if [ -n "$SLURM_QOS" ]; then
    SBATCH_ARGS+=(
        --qos=${SLURM_QOS} \
    )
fi
# Clusters whose sbatch rejects --gpus-per-node set skip_gres in their cluster yaml.
if [ -z "$SKIP_GRES_ARG" ]; then
    SBATCH_ARGS+=(
        --gpus-per-node=${GPUS_PER_NODE} \
    )
fi
SBATCH_ARGS+=(
    ray.sub
)

# Slurm does not create the --output directory; without it the job dies
# immediately and leaves no log behind.
mkdir -p "${BASE_LOG_DIR}"

# Submit, pulling the job id out of sbatch's "Submitted batch job <id>" line.
JOB_ID=$(sbatch ${SBATCH_ARGS[@]} | awk '{print $4}')

echo "Submitted batch job ${JOB_ID}"

if [ "$1" == "-i" ]; then
    echo "Please run \"bash $(pwd)/${JOB_ID}-attach.sh\" on the cluster to attach to the job."
fi
