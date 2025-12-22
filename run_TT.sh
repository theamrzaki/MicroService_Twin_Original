#!/bin/bash

FREQ_DOMAINS=(
 #"iTransformer"
 # "FreTS"
 # "DLinear"
 # "encoder_decoder"
 # "TimesNet"
 # "FEDformerModel"
 # "FITS_Legendre" OrEdge
 # "Eadro"  
 "AnoFusion"
)

SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source=SN
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --random_seed="$SEED" \
        --data_source=SN
    fi

  done
done



SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source=TT
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --random_seed="$SEED" \
        --data_source=TT
    fi

  done
done
