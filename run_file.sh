# Run the whole pipeline 
# This pipeline is to running original work of author 6 times and record the result of improved embedding model 

#!/bin/bash
num_iteration=$1

for i in $(seq 0 $num_iteration)
do 
    # Step 1: Get embeddings from original embedding model
    # if i == 0 --> run with original embedding model 
    # if i != 0 --> run with finetuned embedding model of iteration i-1    
    bash perspective/2_finetune/scripts/get_embedding.sh $i

    # Step 2: Sample triplets
    bash perspective/1_predict_triplet/scripts/triplet_sampling.sh $i

    # Step 3: Predict triplets
    bash perspective/1_predict_triplet/scripts/predict_triplet.sh $i

    # Step 4: Convert triplets
    bash perspective/2_finetune/scripts/convert_triplet.sh

    # Step 5: 
    bash perspective/2_finetune/scripts/finetune.sh $i
done






