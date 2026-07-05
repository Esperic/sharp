#!/usr/bin/env bash
set -euo pipefail

: "${DATA:?Set DATA=/path/to/av2_data_root/sharp_processed}"
: "${GMP:?Set GMP=anchors/gmp/av2_gmp_xy_k6_t100.npz}"
: "${BASE_CKPT:?Set BASE_CKPT=/path/to/sharp_baseline.ckpt}"

SEED="${SEED:-2333}"

python train.py \
  datamodule.pl_module.data_root="$DATA" \
  model=Sharp_av2 \
  model.pl_module.model.use_gmp=false \
  model.pl_module.model.use_drift_loss=false \
  model.pl_module.model.use_mdf=false \
  model.pl_module.model.use_endpoint_diversity=false \
  tag=sharp_baseline_repro \
  seed="$SEED"

python train.py \
  datamodule.pl_module.data_root="$DATA" \
  checkpoint="$BASE_CKPT" \
  model=Sharp_av2 \
  model.pl_module.model.use_gmp=true \
  model.pl_module.model.gmp_path="$GMP" \
  model.pl_module.model.gmp_sampling=query_aligned \
  model.pl_module.model.gmp_train_noise_scale=1.0 \
  model.pl_module.model.gmp_eval_noise_scale=0.0 \
  model.pl_module.model.gmp_condition_query=true \
  model.pl_module.model.gmp_condition_pi=true \
  model.pl_module.model.gmp_condition_memory=false \
  model.pl_module.model.use_drift_loss=false \
  model.pl_module.model.use_mdf=false \
  model.pl_module.model.use_endpoint_diversity=false \
  tag=drifttraj_gmp_only \
  seed="$SEED"

python train.py \
  datamodule.pl_module.data_root="$DATA" \
  checkpoint="$BASE_CKPT" \
  model=Sharp_av2 \
  model.pl_module.model.use_gmp=true \
  model.pl_module.model.gmp_path="$GMP" \
  model.pl_module.model.use_endpoint_diversity=true \
  model.pl_module.model.diversity_weight=0.03 \
  model.pl_module.model.diversity_sigma=2.0 \
  model.pl_module.model.diversity_warmup_epochs=3 \
  model.pl_module.model.use_drift_loss=false \
  model.pl_module.model.use_mdf=false \
  tag=drifttraj_gmp_div \
  seed="$SEED"

python train.py \
  datamodule.pl_module.data_root="$DATA" \
  checkpoint="$BASE_CKPT" \
  model=Sharp_av2 \
  model.pl_module.model.use_gmp=true \
  model.pl_module.model.gmp_path="$GMP" \
  model.pl_module.model.use_endpoint_diversity=true \
  model.pl_module.model.diversity_weight=0.03 \
  model.pl_module.model.use_drift_loss=true \
  model.pl_module.model.use_mdf=false \
  model.pl_module.model.drift_weight=0.1 \
  model.pl_module.model.drift_warmup_epochs=5 \
  model.pl_module.model.drift_single_radius=0.1 \
  model.pl_module.model.winner_metric=ade_fde \
  model.pl_module.model.winner_fde_weight=1.0 \
  tag=drifttraj_gmp_drift_singleR \
  seed="$SEED"

python train.py \
  datamodule.pl_module.data_root="$DATA" \
  checkpoint="$BASE_CKPT" \
  model=Sharp_av2 \
  model.pl_module.model.use_gmp=true \
  model.pl_module.model.gmp_path="$GMP" \
  model.pl_module.model.gmp_sampling=query_aligned \
  model.pl_module.model.gmp_train_noise_scale=1.0 \
  model.pl_module.model.gmp_eval_noise_scale=0.0 \
  model.pl_module.model.gmp_condition_query=true \
  model.pl_module.model.gmp_condition_pi=true \
  model.pl_module.model.gmp_condition_memory=false \
  model.pl_module.model.use_drift_loss=true \
  model.pl_module.model.use_mdf=true \
  model.pl_module.model.mdf_r_list='[0.02,0.1,0.5]' \
  model.pl_module.model.drift_weight=0.2 \
  model.pl_module.model.drift_warmup_epochs=5 \
  model.pl_module.model.drift_force_clip=1.0 \
  model.pl_module.model.use_endpoint_diversity=true \
  model.pl_module.model.diversity_weight=0.03 \
  model.pl_module.model.diversity_sigma=2.0 \
  model.pl_module.model.use_label_smoothing_ce=true \
  model.pl_module.model.label_smoothing=0.05 \
  model.pl_module.model.winner_metric=ade_fde \
  model.pl_module.model.winner_fde_weight=1.0 \
  tag=drifttraj_full \
  seed="$SEED"
