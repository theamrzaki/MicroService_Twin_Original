#!/bin/bash

FREQ_DOMAIN="FITS_Legendre" 


SEEDS=(1)
# OrAnomaly / without TexFilter (LPF) / without time-freq fredf loss
for SEED in "${SEEDS[@]}"; do

      # OrAnomaly (ours)
      echo "================================================================="
      echo "Running OrAnomaly (ours)"
      echo "================================================================="
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source=TT \
        --experiment_name="RQ2_ablations_components" 

      # Ablation: without TexFilter (LPF)    
      echo "================================================================="
      echo "Running ablation: without TexFilter (LPF)"  
      echo "================================================================="
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source=TT \
        --filter_used="LPF" \
        --experiment_name="RQ2_ablations_components" 


      # Ablation: without time-freq (remove time dimension lambda) fredf loss
      echo "================================================================="
      echo "Running ablation: without time-freq (remove time dimension lambda) fredf loss"
      echo "================================================================="
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --random_seed="$SEED" \
        --data_source=TT \
        --experiment_name="RQ2_ablations_components" \
        --rec_lambda=0.0 --auxi_lambda=1.0
        
      # Ablation: without linear attention 
      echo "================================================================="
      echo "Running ablation: without linear attention"
      echo "================================================================="
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source=TT \
        --modules_attn="no_attn" \
        --experiment_name="RQ2_ablations_components" 
done
