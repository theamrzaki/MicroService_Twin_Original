#!/bin/bash

FREQ_DOMAINS=(
 #"iTransformer"
 # "FreTS"
 # "DLinear"
 # "encoder_decoder"
 # "TimesNet"
 # "FEDformerModel"
#"FITS_Legendre" #OrEdge
 # "Eadro"  
 #"AnoFusion"
 "Art"
)

SEEDS=(1)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source=ART
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --random_seed="$SEED" \
        --data_source=ART
    fi

  done
done
