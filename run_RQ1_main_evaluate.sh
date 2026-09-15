#!/bin/bash



FREQ_DOMAINS=(
#"Eadro"  
####
#"FITS_Legendre" #OrEdge
###
#"AnoFusion"
#"encoder_decoder"
#"Art"
#"Medicine"
"DeepHunt"
)
ORANOMALY_MODELS=("FITS_Legendre" "FreTS" "DLinear" "iTransformer" "FEDformerModel" "FITS_LENGDRE_parallel_oth_compoenents")

DATA_SOURCES=("MSDS") #"SN" "MSDS" "TT" 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

SEEDS=(1) # 2 3
EPOCHS=300
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
        --model_path="./result/${FREQ}-${datasource}-Epochs${EPOCHS}-Seed${SEED}ExpNameRQ1_main-lowshow-simplegraph" \
        --evaluate=true \
        --gpu=true \
        --experiment_name="RQ1_main_Evaluate"
    else
      echo "--------------------------------"
      echo "2) Running $FREQ with Normal-Recreation loss, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Normal-Recreation' \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --model_path="./result/${FREQ}-${datasource}-Epochs${EPOCHS}-Seed${SEED}ExpNameRQ1_main-lowshow-simplegraph" \
        --evaluate=true \
        --gpu=true \
        --experiment_name="RQ1_main_Evaluate"
    fi
  done
  done
done

