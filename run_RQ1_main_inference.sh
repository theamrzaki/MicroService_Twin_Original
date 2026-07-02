#!/bin/bash



FREQ_DOMAINS=(
# "Eadro"  
##
"FITS_Legendre" #OrEdge
##
#"AnoFusion"
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


#SEEDS=(1 2 3)
#
#for SEED in "${SEEDS[@]}"; do
#    for FREQ in "${FREQ_DOMAINS[@]}"; do
#  
#    if [[ " ${ORANOMALY_MODELS[@]} " =~ " ${FREQ} " ]]; then
#      /bin/python3 main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Legendre-style' \
#        --filter_used="LPF" \
#        --random_seed="$SEED" \
#        --data_source=MSDS \
#        --experiment_name="RQ1_main"
#    else
#      /bin/python3 main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Normal-Recreation' \
#        --random_seed="$SEED" \
#        --data_source=MSDS \
#        --experiment_name="RQ1_main"
#    fi
#
#  done
#done


#SEEDS=(1 2 3)
#
#for SEED in "${SEEDS[@]}"; do
#    for FREQ in "${FREQ_DOMAINS[@]}"; do
#  
#    if [[ " ${ORANOMALY_MODELS[@]} " =~ " ${FREQ} " ]]; then
#      /bin/python3 main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Legendre-style' \
#        --filter_used="LPF" \
#        --random_seed="$SEED" \
#        --data_source=SN \
#        --experiment_name="RQ1_main" 
#    else
#      /bin/python3 main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Normal-Recreation' \
#        --random_seed="$SEED" \
#        --data_source=SN \
#        --experiment_name="RQ1_main"
#    fi
#
#  done
#done

source ~/miniconda3/etc/profile.d/conda.sh
# or:
# source ~/miniconda3/etc/profile.d/conda.sh

conda activate RCAEval

SEEDS=(1)

for SEED in "${SEEDS[@]}"; do
    for FREQ in "${FREQ_DOMAINS[@]}"; do
  
    if [[ " ${ORANOMALY_MODELS[@]} " =~ " ${FREQ} " ]]; then
      echo "--------------------------------"
      echo "1) Running $FREQ with Legendre-style loss and LPF filter, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Legendre-style' \
        --filter_used="LPF" \
        --random_seed="$SEED" \
        --data_source=TT \
        --epochs=10 \
        --evaluate=true \
        --model_path="./result/$FREQ-TT-$SEED" \
        --experiment_name="RQ1_main"
    else
      echo "--------------------------------"
      echo "2) Running $FREQ with Normal-Recreation loss, seed $SEED"
      echo "--------------------------------"
      python main.py \
        --FREQ_DOMAIN="$FREQ" \
        --req_loss_approach='Normal-Recreation' \
        --random_seed="$SEED" \
        --data_source=TT \
        --epochs=10 \
        --evaluate=true \
        --experiment_name="RQ1_main"
    fi

  done
done



#for SEED in "${SEEDS[@]}"; do
#    for FREQ in "${FREQ_DOMAINS[@]}"; do
#  
#    if [[ " ${ORANOMALY_MODELS[@]} " =~ " ${FREQ} " ]]; then
#      python main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Legendre-style' \
#        --filter_used="LPF" \
#        --random_seed="$SEED" \
#        --data_source=SN \
#        --experiment_name="RQ1_main"
#    else
#      python main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Normal-Recreation' \
#        --random_seed="$SEED" \
#        --data_source=SN \
#        --experiment_name="RQ1_main"
#    fi
#
#  done
#done
#
#
#
#
#
#for SEED in "${SEEDS[@]}"; do
#    for FREQ in "${FREQ_DOMAINS[@]}"; do
#  
#    if [[ " ${ORANOMALY_MODELS[@]} " =~ " ${FREQ} " ]]; then
#      python main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Legendre-style' \
#        --filter_used="LPF" \
#        --random_seed="$SEED" \
#        --data_source=MSDS \
#        --experiment_name="RQ1_main"
#    else
#      python main.py \
#        --FREQ_DOMAIN="$FREQ" \
#        --req_loss_approach='Normal-Recreation' \
#        --random_seed="$SEED" \
#        --data_source=MSDS \
#        --experiment_name="RQ1_main"
#    fi
#
#  done
#done
#