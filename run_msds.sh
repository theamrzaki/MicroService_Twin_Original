#!/bin/bash

FREQ_DOMAINS=(
   "Eadro"  
 "AnoFusion"
 "Art"
"encoder_decoder"

"FITS_Legendre" #OrEdge

"iTransformer"
 "FreTS"
 "DLinear"
 "TimesNet"
 "FEDformerModel"


)

SEEDS=(1 2 3)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source=MSDS
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --random_seed="$SEED" \
        --data_source=MSDS
    fi

  done
done
