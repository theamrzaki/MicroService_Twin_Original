#!/bin/bash

FREQ_DOMAINS=(
  "FITS_Legendre"  
  #"Eadro"
)

SEEDS=(1)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --case_study=True \
        --epochs=300 \
        --data_source=MSDS
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --random_seed="$SEED" \
        --case_study=True \
        --epochs=300 \
        --data_source=MSDS
    fi

  done
done

#chmod +x run_RQ2_ablations.sh