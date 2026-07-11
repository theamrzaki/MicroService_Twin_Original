#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

FREQ_DOMAIN="FITS_Legendre" 
DATA_SOURCES=("SN" "TT" "MSDS")
SEEDS=(1)
# OrAnomaly / without LPF / without time-freq fredf loss
for SEED in "${SEEDS[@]}"; do
    for data_source in "${DATA_SOURCES[@]}"; do

      # OrAnomaly (ours)
      echo "================================================================="
      echo "Running ablation: without LPF (with linear attention, with FreDF loss), data source: $data_source"  
      echo "================================================================="
      python main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --filter_used="LPF" \
        --gpu=true \
        --experiment_name="RQ2_ablations_components" 
            #
      #
      # Ablation: LPF + no linear attention
      echo "================================================================="
      echo "Running ablation: no linear attention (with LPF, with FreDF loss), data source: $data_source"
      echo "================================================================="
      python main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --modules_attn="no_attn" \
        --filter_used="LPF" \
        --gpu=true \
        --experiment_name="RQ2_ablations_components" 

      #
      # Ablation: without time-freq (remove time dimension lambda) fredf loss
      echo "================================================================="
      echo "Running ablation: without FreDF [remove time dimension lambda] (with linear attention and LPF), data source: $data_source"
      echo "================================================================="
      python main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --experiment_name="RQ2_ablations_components" \
        --filter_used="LPF" \
        --gpu=true \
        --rec_lambda=1.0 --auxi_lambda=0.0
      #
      #
      echo "================================================================="
      echo "Running ablation: No filter applied (with linear attention, with FreDF loss), data source: $data_source"
      echo "================================================================="
      python main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --experiment_name="RQ2_ablations_components" \
        --gpu=true \
        --filter_used="nofilter" 


      echo "================================================================="
      echo "Running ablation: No normlin applied (with linear attention, with FreDF loss, with LPF), data source: $data_source"
      echo "================================================================="
      python main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --experiment_name="RQ2_ablations_components" \
        --filter_used="LPF" \
        --gpu=true \
        --use_normlin=False 
    done
done




FREQ_DOMAIN="FITS_Legendre"
req_loss_approach="Legendre-style"

basis_type=("chebyshev" "fourier" "hermite" "laguerre" "legendre" )


for SEED in "${SEEDS[@]}"; do
    for data in "${DATA_SOURCES[@]}"; do
        for basis in "${basis_type[@]}"; do

        echo "================================================================="
        echo "Running OrAnomaly with basis: $basis and data source: $data"
        echo "================================================================="

        if [ "$basis" == "fourier" ]; then
            req_loss_approach="FreDF-style"
        else
            req_loss_approach="Legendre-style"
        fi

        python main.py \
            --FREQ_DOMAIN="$FREQ_DOMAIN" \
            --req_loss_approach="$req_loss_approach" \
            --random_seed="$SEED" \
            --data_source="$data" \
            --basis_type="$basis" \
            --gpu=true \
            --experiment_name="RQ2_basis_comparison" \
            --filter_used="LPF"
        done
    done
done





FREQ_DOMAINS=(
  "FreTS"
  "DLinear"
  "iTransformer"
  "TimesNet"
  "FEDformerModel"
  "FITS_Legendre" 
)
ORANOMALY_MODELS=("FITS_Legendre" "FreTS" "DLinear" "iTransformer" "FEDformerModel")

for SEED in "${SEEDS[@]}"; do
  for datasource in "${DATA_SOURCES[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do

    if [[ " ${ORANOMALY_MODELS[@]} " =~ " ${FREQ} " ]]; then
      echo "--------------------------------"
      echo "1) Running $FREQ with Legendre-style loss and LPF filter, data source: $datasource"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --evaluate=false \
        --gpu=true \
        --experiment_name="RQ2_architecture"
    else
      echo "--------------------------------"
      echo "2) Running $FREQ with Normal-Recreation loss, data source: $datasource"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Normal-Recreation' \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --evaluate=false \
        --gpu=true \
        --experiment_name="RQ2_architecture"
    fi
  done
  done
done

