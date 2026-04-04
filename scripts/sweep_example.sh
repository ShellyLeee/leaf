#!/usr/bin/env bash
set -e

# Example hyperparameter sweep (manual shell loop)
for hidden_dim in 256 512 1024; do
  for lr in 1e-3 5e-4; do
    python train.py \
      --config cfgs/config.yaml \
      --opts output_dir=./outputs/mlp_hd${hidden_dim}_lr${lr} model.hidden_dim=${hidden_dim} optimizer.lr=${lr}
  done
done
