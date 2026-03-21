#!/bin/bash

FREQ_DOMAINS=(
  #"Art"
  "FITS_Legendre"  
)

SEEDS=(1)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --case_study=True \
        --epochs=5 \
        --experiment_name="RQ3_case_study" \
        --data_source=MSDS
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --random_seed="$SEED" \
        --req_loss_approach='Normal-Recreation' \
        --case_study=True \
        --epochs=5 \
        --experiment_name="RQ3_case_study" \
        --data_source=MSDS
    fi

  done
done

#chmod +x run_RQ2_ablations.sh