# DeepSeek-V4 nested-tree patches

The dsv4 recipe requires changes in the vendored Megatron-LM and
Megatron-Bridge working trees. Those live in nested git repos, so a fresh
clone of this branch does NOT carry them — apply these patches after
checking out the pinned trees:

```bash
# Megatron-LM: dsv4_hybrid attention graft (onto pin 14346b65a).
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM
git checkout 14346b65a
git am ../../../../patches/megatron-lm-dsv4-graft.patch

# Megatron-Bridge: conversion fixes for the V4 RL refit.
cd ..
git am ../../patches/megatron-bridge-dsv4-fixes.patch
```

Contents:
- `megatron-lm-dsv4-graft.patch` — csa.py + deepseek_v4_hybrid_attention.py
  (verbatim from NVIDIA-internal ADLR/megatron-lm `dsv4_mtp` @ 8fd14d526),
  MLA rope fusions, module-spec builder + dispatcher case, config fields
  (csa_*, o_groups, o_lora_rank), pin-API adaptations. Public Megatron-LM
  has no `dsv4_hybrid` experimental attention variant.
- `megatron-bridge-dsv4-fixes.patch` — skip None conversion-task slots in
  the HF->Megatron load loop; apply the hf-keys check in the PP-placeholder
  pass of build_conversion_tasks so task lists agree across PP ranks.

The same commits exist as local branches `linj/kernelwriter-dsv4` in both
nested repos (pushable to forks; the NVIDIA org remotes reject direct push).
