

#!/bin/bash
MODEL_NAMES=("Eadro" "AnoFusion" "encoder_decoder" "FITS_Legendre" "Art")
DATASET_NAMES=("TT") #"MSDS" "SN" 
#IP_ADDRESS="130.63.254.140" #db2003smaller
IP_ADDRESS="130.63.103.80"
SEEDS=("3")
#DEVICE_NAME="db2003smaller"
DEVICE_NAME="db2003larger"

# Give your user permission to read the source folder so you don't need sudo inside the loops
sudo chown -R $(whoami) "/home/db2003/Desktop/Amr/(Journal) MicroService_Twin_Original/result/"

for seed in "${SEEDS[@]}"; do
    for model in "${MODEL_NAMES[@]}"; do
        for dataset in "${DATASET_NAMES[@]}"; do
            if [ "$dataset" == "TT" ]; then
                EPOCHS="50"
            else
                EPOCHS="300"
            fi
            
            # No sudo needed here anymore
            scp -r "/home/db2003/Desktop/Amr/(Journal) MicroService_Twin_Original/result/${model}-${dataset}-Epochs${EPOCHS}-Seed${seed}ExpNameRQ1_main-lowshow-simplegraph" $DEVICE_NAME@$IP_ADDRESS:/home/$DEVICE_NAME/MicroService_Twin_Original/result/
        done
    done
done
