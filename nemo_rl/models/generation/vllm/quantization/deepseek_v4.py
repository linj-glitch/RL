# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Engine-side checkpoint-format requantization for DeepSeek-V4 refit.

Why this exists: vLLM's layerwise reload (armed by NeMo-RL for DeepseekV4
refit) counts a layer as complete only when the number of loaded elements
reaches the total numel of the layer's checkpoint-format parameters —
including the quantized weight *scales* restored from load-time metadata
(``get_layer_size`` in vllm's reload/utils.py). NeMo-RL's refit wire carries
bf16 HF-format tensors only, never scales, so no quantized layer ever
completes: every buffered weight (a view of the multi-GB broadcast bucket)
stays resident until the engine OOMs. Even if completion were forced, the
flush replays weight bytes verbatim, so raw bf16 would be bit-cast into
fp8/MXFP4 containers with stale/garbage scales.

The fix reproduces the *disk checkpoint stream* on the wire. For every
incoming ``<x>.weight`` that the serving checkpoint pairs with ``<x>.scale``,
quantize engine-side into the exact checkpoint byte layout:

- MXFP4 experts (``expert_dtype="fp4"``): e2m1 packed uint8 (two values per
  byte, even element in the low nibble) plus a ue8m0 scale per 32-element
  group along the contraction dim — the layout ``Mxfp4MoEMethod`` /
  MegaMoE containers load from disk.
- Dense / shared-expert / attention linears: blockwise [128, 128] fp8 e4m3
  plus a ue8m0 (power-of-two) inverse scale, matching
  ``DeepseekV4FP8Config``'s ``float8_e8m0fnu`` block-scale parameters.
- FP8-expert checkpoints (``expert_dtype="fp8"``, e.g. Flash-Base): experts
  and dense linears both use blockwise fp8 with float32 scales.

Both the weight and its scale then flow through ``model.load_weights``, so
per-layer completion counting, flushing, and
``process_weights_after_loading`` behave exactly as they do for the initial
disk load: each layer flushes as soon as its weights arrive (~one layer of
staging) and requantization uses the quant method's native semantics.
"""

import json
import os
import re

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_MXFP4_EXPERT_WEIGHT_RE = re.compile(r"\.experts\.\d+\.w[123]\.weight$")

# Passthrough bf16 tensors at or below this size are cloned out of the refit
# bucket's flat broadcast buffer. Buffered weight_loader args hold *views* of
# that buffer, so one small tensor belonging to a layer that does not complete
# within the bucket (e.g. a gate weight waiting on a tensor the trainer never
# sends) would otherwise pin the entire multi-GB bucket until finalize.
# Large passthrough tensors (embed_tokens, lm_head) belong to single-parameter
# layers that complete and flush inside the same load call, so views are safe
# and cloning them would only add transient memory.
_CLONE_PASSTHROUGH_MAX_BYTES = 32 * 1024 * 1024


_MXFP_GROUP_SIZE = 32
_E2M1_MAX = 6.0

# Whether triton_kernels' MXFP4 downcast kernel is usable in this process:
# None = untried, True = usable, False = failed once (never retried). The
# kernel is only ever exercised on SM100 during normal serving (on H100 the
# engine loads pre-packed MXFP4 from disk, so refit is its first call site)
# and its JIT compilation fails on SM90.
_TRITON_MXFP4_DOWNCAST_USABLE: bool | None = None


def _mxfp4_downcast_torch(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch MXFP4 quantization along the last dim (checkpoint layout).

    Byte-identical port of triton_kernels' ``downcast_to_mxfp`` in its
    default ROUND_UP scale mode, using only plain torch ops (no triton JIT,
    which fails to compile on SM90): per 32-element group, the ue8m0 scale
    byte is the exponent of ``2**ceil(log2(absmax / 6))``; values are divided
    by that scale and rounded to the e2m1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    (nearest, ties away from zero) with the sign in nibble bit 3; nibble
    pairs pack as ``even | (odd << 4)``.

    Args:
        weight: High-precision (bf16/fp16/fp32) weight of shape [..., K],
            K divisible by 32.

    Returns:
        Tuple of (packed e2m1 uint8 tensor of shape [..., K/2], ue8m0 scale
        uint8 tensor of shape [..., K/32]).
    """
    k = weight.shape[-1]
    if k % _MXFP_GROUP_SIZE != 0:
        raise ValueError(
            "MXFP4 quantization requires the last dim to be divisible by "
            f"{_MXFP_GROUP_SIZE}, got shape {tuple(weight.shape)}."
        )
    src = weight.reshape(
        *weight.shape[:-1], k // _MXFP_GROUP_SIZE, _MXFP_GROUP_SIZE
    ).float()
    group_max = src.abs().amax(dim=-1, keepdim=True)
    # 2**ceil(log2(absmax / 6)) via fp32 bit twiddling: adding 0x007FFFFF
    # bumps the exponent unless the mantissa is all zeros. All-zero groups
    # keep a zero scale byte: their members quantize to 0x0 nibbles below
    # (quant_scale forced to 0) and upcast multiplies by the decoded 0.0
    # scale, yielding exact zeros without NaN/inf — same as the kernel.
    ds_int = (group_max / _E2M1_MAX).view(torch.int32)
    ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
    dequant_scale = ds_int_rounded.view(torch.float32)
    quant_scale = torch.where(
        dequant_scale == 0, torch.zeros_like(dequant_scale), 1.0 / dequant_scale
    )
    q = (src * quant_scale).contiguous()

    q_int = q.view(torch.int32)
    exponents = (q_int >> 23) & 0xFF
    mantissas = q_int & 0x7FFFFF
    e8_bias, e2_bias = 127, 1
    # Subnormal e2m1 magnitudes (|q| < 1): shift the implicit leading 1 into
    # the mantissa. Shift amounts are clamped to 31 to match PTX shift
    # semantics; larger shifts only arise for magnitudes that round to zero.
    sub_shift = (e8_bias - 1 - exponents).clamp(0, 31)
    sub_mant = (0x400000 | (mantissas >> 1)) >> sub_shift
    mantissas = torch.where(exponents < e8_bias, sub_mant, mantissas)
    exponents = exponents.clamp(min=e8_bias - e2_bias) - (e8_bias - e2_bias)
    # 3-bit magnitude code: exponent and top mantissa bit plus a guard bit,
    # rounded nearest (ties up) and saturated at code 0x7 (= 6.0).
    code = ((((exponents << 2) | (mantissas >> 21)) + 1) >> 1).clamp(max=0x7)
    nibbles = code | (torch.signbit(q).to(torch.int32) << 3)

    nibbles = nibbles.reshape(*weight.shape[:-1], k // 2, 2)
    qweight = (nibbles[..., 0] | (nibbles[..., 1] << 4)).to(torch.uint8)
    scale = (ds_int_rounded >> 23).squeeze(-1).to(torch.uint8)
    return qweight, scale


def _bf16_to_mxfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight to checkpoint-layout MXFP4 along the last dim.

    Tries triton_kernels' downcast kernel once per process for CUDA inputs;
    if it fails (e.g. its JIT does not compile on SM90 engines), permanently
    falls back to the byte-identical pure-torch implementation.

    Args:
        weight: High-precision (bf16/fp16/fp32) weight of shape [..., K],
            K divisible by 32.

    Returns:
        Tuple of (packed e2m1 uint8 tensor of shape [..., K/2] with the even
        element in the low nibble, ue8m0 scale uint8 tensor of shape
        [..., K/32]) — the byte layout DeepSeek-V4 MXFP4 checkpoints store.
    """
    global _TRITON_MXFP4_DOWNCAST_USABLE
    if weight.is_cuda and _TRITON_MXFP4_DOWNCAST_USABLE is not False:
        try:
            from triton_kernels.numerics_details.mxfp import downcast_to_mxfp

            qweight, scale = downcast_to_mxfp(weight, torch.uint8, axis=-1)
        except Exception:
            _TRITON_MXFP4_DOWNCAST_USABLE = False
            logger.warning(
                "triton_kernels' MXFP4 downcast is unusable on this device "
                "(its JIT is known to fail on SM90); using the pure-torch "
                "MXFP4 quantizer for this and all later refit buckets.",
                exc_info=True,
            )
        else:
            _TRITON_MXFP4_DOWNCAST_USABLE = True
            return qweight, scale
    return _mxfp4_downcast_torch(weight)


def _pow2_scale_to_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Encode exact power-of-two float scales as raw float8_e8m0fnu bytes.

    Encoding the byte directly (rather than relying on torch's float32 ->
    float8_e8m0fnu cast kernels) keeps the wire tensor byte-identical to the
    checkpoint's serialized scales.
    """
    exponent = torch.where(
        scale > 0,
        torch.log2(scale.float()),
        torch.full_like(scale, -127.0, dtype=torch.float32),
    )
    byte = (exponent.round().clamp(-127.0, 127.0) + 127.0).to(torch.uint8)
    return byte.view(torch.float8_e8m0fnu)


def _bf16_to_fp8_block(
    weight: torch.Tensor, scale_e8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight to blockwise [128, 128] fp8 checkpoint format.

    Args:
        weight: High-precision 2D weight.
        scale_e8m0: If True, ceil scales to powers of two and return them as
            float8_e8m0fnu (DeepSeek-V4 fp4-expert checkpoints); otherwise
            return float32 scales (fp8-expert checkpoints).

    Returns:
        Tuple of (fp8 e4m3 weight with ``weight``'s shape, inverse block
        scale of shape [ceil(M/128), ceil(N/128)]).
    """
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    qweight, scale = per_block_cast_to_fp8(
        weight, block_size=[128, 128], use_ue8m0=scale_e8m0
    )
    if scale_e8m0:
        scale = _pow2_scale_to_e8m0(scale)
    return qweight, scale


def apply_attn_sinks(
    model: torch.nn.Module, attn_sinks: dict[str, torch.Tensor]
) -> None:
    """Apply buffered attention-sink weights to the finalized model.

    ``DeepseekV4Model.load_weights`` loads ``attn_sink`` with a direct
    ``param[:n].copy_()`` rather than a ``weight_loader`` call. While the
    layerwise reload is armed, that parameter is restored on the meta device,
    so the copy silently no-ops and the trained sink values would be dropped
    (the stale kernel tensor is placed back at finalize). The refit transform
    therefore diverts attn_sink tensors, and this function applies them after
    ``finalize_layerwise_reload`` has restored the real kernel tensors.

    Args:
        model: The finalized vLLM model (``DeepseekV4ForCausalLM``).
        attn_sinks: Mapping of wire names (e.g. ``layers.3.attn.attn_sink``)
            to full, unsharded sink tensors of shape [num_attention_heads].
    """
    if not attn_sinks:
        return
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    params = dict(model.named_parameters())
    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    for name, weight in attn_sinks.items():
        param_name = name if name in params else f"model.{name}"
        if param_name not in params:
            # The layer lives on another pipeline-parallel rank.
            continue
        param = params[param_name]
        n_local_heads = weight.shape[0] // tp_size
        shard = weight[tp_rank * n_local_heads : (tp_rank + 1) * n_local_heads]
        # The parameter is padded to the platform head count and initialized
        # to -inf; only the first n_local_heads slots hold real sinks.
        param.data[: shard.shape[0]].copy_(shard)


class DeepseekV4CheckpointFormatQuantizer:
    """Requantizes bf16 refit wire tensors into DeepSeek-V4 checkpoint format.

    Which weights carry scales (and therefore need quantization) is read from
    the serving checkpoint's safetensors index: every ``<x>.weight`` paired
    with an ``<x>.scale`` entry is quantized; everything else passes through
    as bf16, exactly mirroring the on-disk stream the engine loaded at init.
    """

    def __init__(self, model_path: str, expert_dtype: str) -> None:
        """Initializes the quantizer from the serving checkpoint layout.

        Args:
            model_path: Local path of the checkpoint the engine serves
                (must contain ``model.safetensors.index.json``).
            expert_dtype: ``"fp4"`` (MXFP4 experts, ue8m0 dense scales) or
                ``"fp8"`` (blockwise fp8 experts, float32 scales).
        """
        if expert_dtype not in ("fp4", "fp8"):
            raise ValueError(
                f"Unsupported DeepSeek-V4 expert_dtype={expert_dtype!r}; "
                "expected 'fp4' or 'fp8'."
            )
        self._expert_dtype = expert_dtype
        # fp4-expert checkpoints serialize all fp8 linear scales as ue8m0
        # (float8_e8m0fnu); fp8-expert checkpoints use float32.
        self._scale_e8m0 = expert_dtype == "fp4"

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        try:
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
        except OSError as e:
            raise RuntimeError(
                "DeepSeek-V4 layerwise refit requires the serving "
                f"checkpoint's safetensors index at {index_path} to identify "
                "which weights carry quantization scales."
            ) from e
        self._scale_names = {
            name for name in weight_map if name.endswith(".scale")
        }
        logger.info(
            "DeepSeek-V4 refit quantizer: %d scaled weights in checkpoint, "
            "expert_dtype=%s",
            len(self._scale_names),
            expert_dtype,
        )

    def transform(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> tuple[list[tuple[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """Transforms one bucket of bf16 wire tensors into checkpoint format.

        Args:
            weights: List of (HF checkpoint name, bf16 tensor) pairs as
                broadcast by the trainer.

        Returns:
            Tuple of (transformed list where each quantized weight is
            followed by its ``<x>.scale`` entry, dict of diverted attn_sink
            tensors to be applied post-finalize via ``apply_attn_sinks``).
        """
        out: list[tuple[str, torch.Tensor]] = []
        attn_sinks: dict[str, torch.Tensor] = {}
        for name, weight in weights:
            # MTP weights are consumed by the separate drafter refit path;
            # the main model's loader skips them.
            if name.startswith("mtp.") or ".mtp." in name:
                out.append((name, weight))
                continue
            if name.endswith(".attn_sink"):
                # Clone: the buffered copy must outlive this bucket's flat
                # broadcast buffer (applied only at finalize).
                attn_sinks[name] = weight.detach().clone()
                continue
            scale_name = (
                name[: -len("weight")] + "scale"
                if name.endswith(".weight")
                else None
            )
            if scale_name is None or scale_name not in self._scale_names:
                if (
                    isinstance(weight, torch.Tensor)
                    and weight.nbytes <= _CLONE_PASSTHROUGH_MAX_BYTES
                ):
                    weight = weight.detach().clone()
                out.append((name, weight))
                continue
            if self._expert_dtype == "fp4" and _MXFP4_EXPERT_WEIGHT_RE.search(
                name
            ):
                qweight, scale = _bf16_to_mxfp4(weight)
            else:
                if weight.dim() != 2:
                    raise ValueError(
                        f"Expected a 2D weight for blockwise fp8 refit "
                        f"quantization, got {name} with shape "
                        f"{tuple(weight.shape)}."
                    )
                qweight, scale = _bf16_to_fp8_block(weight, self._scale_e8m0)
            out.append((name, qweight))
            out.append((scale_name, scale))
        return out, attn_sinks
