#!/bin/bash



FREQ_DOMAINS=(
"Eadro"  
##
"FITS_Legendre" #OrEdge
##
"AnoFusion"
"encoder_decoder"
"Art"
)
ORANOMALY_MODELS=("FITS_Legendre" "FreTS" "DLinear" "iTransformer" "FEDformerModel" "FITS_LENGDRE_parallel_oth_compoenents")

DATA_SOURCES=("SN" "MSDS" "TT")

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

SEEDS=(1)

for SEED in "${SEEDS[@]}"; do
  for datasource in "${DATA_SOURCES[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do

    if [[ " ${ORANOMALY_MODELS[@]} " =~ " ${FREQ} " ]]; then
      echo "--------------------------------"
      echo "1) Running $FREQ with Legendre-style loss and LPF filter, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --evaluate=false \
        --gpu=true \
        --experiment_name="RQ1_main"
    else
      echo "--------------------------------"
      echo "2) Running $FREQ with Normal-Recreation loss, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Normal-Recreation' \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --evaluate=false \
        --gpu=true \
        --experiment_name="RQ1_main"
    fi
  done
  done
done


SEEDS=(1)
auxi_lambdas=(0.2 0.5 1.0)
for SEED in "${SEEDS[@]}"; do
  for datasource in "${DATA_SOURCES[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
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
        --experiment_name="RQ3_sensitivity"
      done
  done
  done
done

linear_attn_dim=(5 10 20)
for SEED in "${SEEDS[@]}"; do
  for datasource in "${DATA_SOURCES[@]}"; do
      for dim in "${linear_attn_dim[@]}"; do
      echo "--------------------------------"
      echo "1) Running Linear attn of $dim with Legendre-style loss and LPF filter, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --evaluate=false \
        --gpu=true \
        --linear_attn_dim="$dim" \
        --experiment_name="RQ3_sensitivity"
      done
  done
done