#!/bin/bash

FREQ_DOMAIN="FITS_Legendre" 
data_source="TT"

SEEDS=(1)
# OrAnomaly / without TexFilter (LPF) / without time-freq fredf loss
for SEED in "${SEEDS[@]}"; do

      # OrAnomaly (ours)
      echo "================================================================="
      echo "Running ablation: without TexFilter (LPF)"  
      echo "================================================================="
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --req_loss_approach='Legendre-style' \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --filter_used="LPF" \
        --experiment_name="RQ2_ablations_components" 
            #
      #
      # Ablation: LPF + no linear attention
      echo "================================================================="
      echo "Running ablation: LPF + no linear attention"
      echo "================================================================="
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --modules_attn="no_attn" \
        --filter_used="LPF" \
        --experiment_name="RQ2_ablations_components" 
        
      #
      # Ablation: without time-freq (remove time dimension lambda) fredf loss
      echo "================================================================="
      echo "Running ablation: LPF + without time-freq (remove time dimension lambda) fredf loss"
      echo "================================================================="
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --experiment_name="RQ2_ablations_components" \
        --filter_used="LPF" \
        --rec_lambda=1.0 --auxi_lambda=0.0


      echo "================================================================="
      echo "Running ablation: No filter applied"
      echo "================================================================="
      /bin/python3 main.py \
        --FREQ_DOMAIN="$FREQ_DOMAIN" \
        --random_seed="$SEED" \
        --data_source="$data_source" \
        --experiment_name="RQ2_ablations_components" \
        --filter_used="nofilter" 
done



#!/bin/bash

FREQ_DOMAIN="FITS_Legendre" 
req_loss_approach="Legendre-style"

basis_type=("chebyshev" "fourier" "hermite" "laguerre" "legendre" )
SEEDS=(1)
#"fourier"
# Run OrAnomaly with different basis types (both in the projection and FreDF loss)
for SEED in "${SEEDS[@]}"; do
      for basis in "${basis_type[@]}"; do
        echo "================================================================="
        echo "Running OrAnomaly with basis: $basis"
        echo "================================================================="
        /bin/python3 main.py \
          --FREQ_DOMAIN="$FREQ_DOMAIN" \
          --req_loss_approach="$req_loss_approach" \
          --random_seed="$SEED" \
          --data_source=TT \
          --basis_type="$basis" \
          --experiment_name="RQ2_basis_comparison"  \
          --filter_used="LPF" \

      done
done