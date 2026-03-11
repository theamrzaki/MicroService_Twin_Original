#!/bin/bash

FREQ_DOMAINS=(
#  "Eadro"  
#"AnoFusion"
#"encoder_decoder"
#"Art"

"FITS_Legendre" #OrEdge

#"iTransformer"
# "TimesNet"
#  "FEDformerModel"
# "FreTS"
# "DLinear"


)




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
