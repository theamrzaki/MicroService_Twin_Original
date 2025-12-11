#!/bin/bash

FREQ_DOMAINS=(
  "FITS_Legendre"
  "iTransformer"
  "DLinear"
  "TimesNet"
  "FreTS"
  "FEDformerModel"
  "encoder_decoder"
)

SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source=SE
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --random_seed="$SEED" \
        --data_source=SE
    fi

  done
done

