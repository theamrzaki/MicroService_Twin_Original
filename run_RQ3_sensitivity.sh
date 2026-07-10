#!/bin/bash



FREQ_DOMAINS=(
"FITS_Legendre"
)
ORANOMALY_MODELS=("FITS_Legendre")

DATA_SOURCES=("MSDS" "TT") # "MSDS" "TT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

SEEDS=(1 2 3)
#degrees=(1 7)
#for SEED in "${SEEDS[@]}"; do
#  for datasource in "${DATA_SOURCES[@]}"; do
#    for FREQ in "${FREQ_DOMAINS[@]}"; do
#      for degree in "${degrees[@]}"; do
#      echo "--------------------------------"
#      echo "1) Running $FREQ with Legendre-style loss and LPF filter, seed $SEED"
#      echo "--------------------------------"
#      python main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Legendre-style' \
#        --filter_used="LPF" \
#        --random_seed="$SEED" \
#        --data_source="$datasource" \
#        --evaluate=false \
#        --gpu=true \
#        --degree="$degree" \
#        --experiment_name="RQ3_sensitivity"
#      done
#  done
#  done
#done




auxi_lambdas=(0.0 0.01 0.2 0.5 1.0)
for SEED in "${SEEDS[@]}"; do
  for datasource in "${DATA_SOURCES[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
      for a in "${auxi_lambdas[@]}"; do
      echo "--------------------------------"
      echo "1) Running $a with Legendre-style loss and LPF filter, seed $SEED"
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
        --experiment_name="RQ3_sensitivity"
      done
  done
  done
done