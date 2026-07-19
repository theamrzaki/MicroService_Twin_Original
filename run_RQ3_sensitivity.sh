#!/bin/bash



FREQ=(
"FITS_Legendre"
)

DATA_SOURCES=("SN" "MSDS" "TT") # "MSDS" "TT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

SEEDS=(2)
auxi_lambdas=(0.0 0.2 1.0) #0.01 0.5
for SEED in "${SEEDS[@]}"; do
  for datasource in "${DATA_SOURCES[@]}"; do
      for a in "${auxi_lambdas[@]}"; do
      echo "--------------------------------"
      echo "1) Running auxi_lambdas of $a with Legendre-style loss and LPF filter, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --evaluate=false \
        --gpu=true \
        --auxi_lambda="$a" \
        --experiment_name="RQ3_sensitivity_auxi_lambda"
      done
  done
done





#linear_attn_dim=(5 10 20)
#for SEED in "${SEEDS[@]}"; do
#  for datasource in "${DATA_SOURCES[@]}"; do
#      for dim in "${linear_attn_dim[@]}"; do
#      echo "--------------------------------"
#      echo "1) Running Linear attn of $dim with Legendre-style loss and LPF filter, seed $SEED"
#      echo "--------------------------------"
#      python main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Legendre-style' \
#        --filter_used="LPF" \
#        --random_seed="$SEED" \
#        --data_source="$datasource" \
#        --evaluate=false \
#        --gpu=true \
#        --linear_attn_dim="$dim" \
#        --experiment_name="RQ3_sensitivity"
#      done
#  done
#done