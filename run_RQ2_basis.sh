#!/bin/bash

FREQ_DOMAIN="FITS_Legendre" 
req_loss_approach="Legendre-style"

basis_type=("legendre" "chebyshev"   "hermite" "laguerre")
SEEDS=(1 2 3)
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
          --experiment_name="RQ2_basis_comparison" 
      done
done
#use chmod +x run_RQ2_basis.sh to make it executable
