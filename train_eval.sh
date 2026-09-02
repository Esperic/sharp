#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train_eval.sh TAG [HYDRA_OVERRIDE ...]
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# == 0 )); then
  echo "Usage: $0 TAG [HYDRA_OVERRIDE ...]" >&2
  exit 2
fi

data_root=/mnt/datasets/av2_motion_forecasting/sharp_processed
tag=$1
shift
run_dir="$PWD/outputs/$tag/$(date +%Y%m%d-%H%M%S)"

export CUDA_VISIBLE_DEVICES=0,1,2,3

python train.py \
  datamodule.pl_module.data_root="$data_root" \
  gpus=4 \
  "$@" \
  wandb=online \
  tag="$tag" \
  output_dir="$run_dir"

checkpoint="$({
  find "$run_dir/checkpoints" -maxdepth 1 -type f -name '*.ckpt' -printf '%f\t%p\n'
} | LC_ALL=C sort -t_ -k3,3g | head -n1 | cut -f2-)"

test -n "$checkpoint"
echo "Evaluating best checkpoint: $checkpoint"

python eval.py \
  datamodule.pl_module.data_root="$data_root" \
  gpus=4 \
  "$@" \
  checkpoint="$checkpoint"
