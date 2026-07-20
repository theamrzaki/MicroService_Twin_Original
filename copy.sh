

#!/bin/bash
MODEL_NAMES=("encoder_decoder")
DATASET_NAMES=("MSDS" "SN" "TT")
IP_ADDRESS="130.63.254.162" #db2003smaller
#IP_ADDRESS="130.63.103.80"
SEEDS=("1")
DEVICE_NAME="db2003smaller"

#This sets up SSH keys so scp never asks for a password again.
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
ssh-copy-id $DEVICE_NAME@$IP_ADDRESS

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
