
FREQ_DOMAIN="FITS_Legendre"
req_loss_approach="Legendre-style"
DATA_SOURCES=("SN" "MSDS" "TT") # "MSDS" "SN" "TT" 

basis_type=("fourier" "hermite" "laguerre" "legendre" "chebyshev") 
degree=("3" "7" "9")
SEEDS=(1)

for SEED in "${SEEDS[@]}"; do
    for data in "${DATA_SOURCES[@]}"; do
        for deg in "${degree[@]}"; do
            for basis in "${basis_type[@]}"; do

            echo "================================================================="
            echo "Running OrAnomaly with basis: $basis | data: $data | degree: $deg | seed: $SEED"
            echo "================================================================="

            #skip if SN fourier seed 1 degree 3 (already done)
            if [ "$data" == "SN" ] && [ "$basis" == "fourier" ] && [ "$deg" == "3" ] && [ "$SEED" == "1" ]; then
                echo "Skipping SN fourier degree 3 seed 1 (already done)"
                continue
            fi

            if [ "$basis" == "fourier" ]; then
                req_loss_approach="FreDF-style"
            else
                req_loss_approach="Legendre-style"
            fi

            python main.py \
                --FREQ_DOMAIN="$FREQ_DOMAIN" \
                --req_loss_approach="$req_loss_approach" \
                --random_seed="$SEED" \
                --data_source="$data" \
                --basis_type="$basis" \
                --gpu=true \
                --degree="$deg" \
                --experiment_name="RQ2z5_basis_degree_comparison" \
                --filter_used="LPF"
            done
        done
    done
done

