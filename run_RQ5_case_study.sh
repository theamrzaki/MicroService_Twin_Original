#!/bin/bash

FREQ_DOMAINS=(
  #"Eadro"
  #"AnoFusion"
  #"encoder_decoder"
  "FITS_Legendre"  #OrEdge
)

SEEDS=(1)
datasource="MSDS"
EPOCHS=300
SEED=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RCAEval

for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --case_study=True \
        --evaluate=true \
        --model_path="./result/${FREQ}-${datasource}-Epochs${EPOCHS}-Seed${SEED}ExpNameRQ1_main-lowshow-simplegraph" \
        --experiment_name="RQ3_case_study" \
        --data_source=$datasource
    else
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --random_seed="$SEED" \
        --req_loss_approach='Normal-Recreation' \
        --case_study=True \
        --experiment_name="RQ3_case_study" \
        --evaluate=true \
        --model_path="./result/${FREQ}-${datasource}-Epochs${EPOCHS}-Seed${SEED}ExpNameRQ1_main-lowshow-simplegraph" \
        --experiment_name="RQ3_case_study" \
        --data_source=$datasource
    fi

done

#chmod +x run_RQ2_ablations.sh