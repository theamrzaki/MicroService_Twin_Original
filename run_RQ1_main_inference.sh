#!/bin/bash



FREQ_DOMAINS=(
##
"FITS_Legendre" #OrEdge
##
"AnoFusion"
"Eadro"  
#"encoder_decoder"
#"Art"
##
##
# "FreTS"
# "DLinear"
#"iTransformer"
 #"TimesNet"
 # "FEDformerModel"


)
ORANOMALY_MODELS=("FITS_Legendre" "FreTS" "DLinear" "iTransformer" "FEDformerModel" "FITS_LENGDRE_parallel_oth_compoenents")
DATA_SOURCES=("TT") # "SN") MSDS TT

# as miniforge is installed on Raspberry Pi
source ~/miniforge3/etc/profile.d/conda.sh
conda activate RCAEval

SEEDS=(2 3)

for SEED in "${SEEDS[@]}"; do
  for datasource in "${DATA_SOURCES[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
    #if TT --> epochs 50 else epochs 300
    if [[ "$datasource" == "TT" ]]; then
      EPOCHS=50
    else
      EPOCHS=300
    fi
    if [[ " ${ORANOMALY_MODELS[@]} " =~ " ${FREQ} " ]]; then
      echo "--------------------------------"
      echo "1) Running $FREQ with Legendre-style loss and LPF filter, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --evaluate=true \
        --gpu=false\
        --model_path="./result/${FREQ}-${datasource}-Epochs${EPOCHS}-Seed${SEED}ExpNameRQ1_main" \
        --experiment_name="RQ1_main_Edge"
    else
      echo "--------------------------------"
      echo "2) Running $FREQ with Normal-Recreation loss, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Normal-Recreation' \
        --random_seed="$SEED" \
        --data_source="$datasource" \
        --gpu=false \
        --evaluate=true \
        --model_path="./result/$FREQ-$datasource-Epochs$EPOCHS-Seed$SEED" \
        --experiment_name="RQ1_main_Edge"
    fi
  done
  done
done
