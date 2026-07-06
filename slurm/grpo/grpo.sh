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

# Optional settings for CudaGym (colocated / disjoint / remote hosting modes)
export CUDAGYM_MODE=${CUDAGYM_MODE:-DEFAULT_CUDAGYM_MODE}
export CUDAGYM_ENABLED=${CUDAGYM_ENABLED:-DEFAULT_CUDAGYM_ENABLED}
export CUDAGYM_CONTAINER=${CUDAGYM_CONTAINER:-DEFAULT_CUDAGYM_CONTAINER}
export ARTIFACTS_DIR=${ARTIFACTS_DIR:-DEFAULT_ARTIFACTS_DIR}
export CCACHE_DIR=${CCACHE_DIR:-DEFAULT_CCACHE_DIR}
export CUDAGYM_NUM_NODES=${CUDAGYM_NUM_NODES:-DEFAULT_CUDAGYM_NUM_NODES}
# remote mode: the eval endpoint URL (Astra/Modal or a hand-started server). Colocated/
# disjoint modes ignore this — ray.sub exports the in-allocation LB URL instead.
export CUDAGYM_UNIFIED_SERVER_URL=${CUDAGYM_UNIFIED_SERVER_URL:-DEFAULT_CUDAGYM_UNIFIED_SERVER_URL}

# API keys
export WANDB_PROJECT="${WANDB_PROJECT:-atlas_nemorl}"
export WANDB_API_KEY=${WANDB_API_KEY:-DEFAULT_WANDB_API_KEY}
export HF_TOKEN=${HF_TOKEN:-DEFAULT_HF_TOKEN}
export CUDAGYM_AUTH_TOKEN=${CUDAGYM_AUTH_TOKEN:-DEFAULT_CUDAGYM_AUTH_TOKEN}

# Slurm allocation options
export NUM_NODES=${NUM_NODES:-DEFAULT_NUM_NODES}
export TIME=${TIME:-DEFAULT_TIME}
export GPUS_PER_NODE=${GPUS_PER_NODE:-DEFAULT_GPUS_PER_NODE}
# =======================================================================

export HF_HOME=${CACHE_PATH}/huggingface
export UV_CACHE_DIR=${CACHE_PATH}/uv
export OUTPUT_DIR=${OUTPUT_ROOT}/${EXP_NAME}

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
#   M0 native   -> examples/run_grpo_cuda.py           + "--extra atlas"
#   M1 nemo-gym -> examples/nemo_gym/run_grpo_nemo_gym.py + "--extra atlas --extra nemo_gym"
# --extra atlas pulls the cudagym SDK (thin client) into the venv; the driver and
# the SYSTEM-venv env actors import cudagym.sdk/cudagym.contracts.
export RUN_SCRIPT=${RUN_SCRIPT:-DEFAULT_RUN_SCRIPT}
export UV_EXTRAS=${UV_EXTRAS:-DEFAULT_UV_EXTRAS}

export COMMAND="uv run ${UV_EXTRAS} ${RUN_SCRIPT} \
    --config examples/configs/recipes/atlas/${CONFIG_NAME} \
    checkpointing.checkpoint_dir=${OUTPUT_DIR}/ckpts \
    logger.log_dir=${OUTPUT_DIR}/logs/wandb \
    logger.wandb.project=${WANDB_PROJECT} \
    logger.wandb.name=${EXP_NAME} \
    logger.wandb_enabled=true \
    cluster.num_nodes=${TRAIN_NUM_NODES} \
    cluster.gpus_per_node=${GPUS_PER_NODE:-8} \
    ${EXTRA_CONFIG_OPTS}
"

cwd=$(pwd -P)
cwd_parent=$(dirname $cwd)

export MOUNTS="$cwd_parent:$cwd_parent,$cwd:/opt/nemo-rl,$WORKSPACE_PATH:$WORKSPACE_PATH,$WORKSPACE_PATH:/cluster_workspace,$MODELS_PATH:/models,$DATASETS_PATH:/datasets"
export PYTHONPATH="$cwd/3rdparty/cudagym/src:${PYTHONPATH}"

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

# Account/partition come from the cluster yaml (submit_grpo.py fills them).
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-DEFAULT_SLURM_ACCOUNT}
export SLURM_PARTITION=${SLURM_PARTITION:-DEFAULT_SLURM_PARTITION}

SBATCH_ARGS=(
    --nodes=${NUM_NODES} \
    --account=${SLURM_ACCOUNT} \
    --job-name=${SLURM_ACCOUNT}-atlas.grpo.${EXP_NAME} \
    --partition=${SLURM_PARTITION} \
    --dependency=singleton \
    --time=${TIME} \
    --output=${BASE_LOG_DIR}/slurm-%j.out \
)
# EOS does not support --gpus-per-node argument
if [ -z "$SKIP_GRES_ARG" ]; then
    SBATCH_ARGS+=(
        --gpus-per-node=${GPUS_PER_NODE} \
    )
fi
SBATCH_ARGS+=(
    ray.sub
)

JOB_ID=$(sbatch ${SBATCH_ARGS[@]} | awk '{print $4}')

echo "Submitted batch job ${JOB_ID}"

if [ "$1" == "-i" ]; then
    echo "Please run \"bash ${BASE_LOG_DIR}/${JOB_ID}/attach.sh\" on the cluster to attach to the job."
fi
