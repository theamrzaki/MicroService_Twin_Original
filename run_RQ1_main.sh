#!/bin/bash

FREQ_DOMAINS=(
 # "Eadro"  

  "FITS_Legendre" #OrEdge

#"AnoFusion"
#"encoder_decoder"
#"Art"
#
#
# "FreTS"
# "DLinear"
#"iTransformer"
# "TimesNet"
#  "FEDformerModel"



)

SEEDS=(2 3)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source=SN \
        --experiment_name="RQ1_main" 
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Normal-Recreation' \
        --random_seed="$SEED" \
        --data_source=SN \
        --experiment_name="RQ1_main"
    fi

  done
done




SEEDS=(2 3)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [ "$FREQ" = "FITS_Legendre" ]; then
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source=MSDS \
        --experiment_name="RQ1_main"
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Normal-Recreation' \
        --random_seed="$SEED" \
        --data_source=MSDS \
        --experiment_name="RQ1_main"
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
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source=TT \
        --experiment_name="RQ1_main"
    else
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Normal-Recreation' \
        --random_seed="$SEED" \
        --data_source=TT \
        --experiment_name="RQ1_main"
    fi

  done
done




