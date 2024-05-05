# Run the whole pipeline 
# This pipeline is to running original work of author 6 times and record the result of improved embedding model 

#!/bin/bash
num_iteration=$1

for i in $(seq 0 $num_iteration)
do 
    # Step 1: Get embeddings from original embedding model
    # if i == 0 --> run with the original embedding model 
    # if i != 0 --> run with the latest finetuned embedding model   
    bash perspective/2_finetune/scripts/get_embedding.sh $i

    # Step 2: Sample triplets
    bash granularity/scripts/sample_pairs.sh $i

    # Step 3: Predict triplets
    bash granularity/scripts/predict_pairs.sh $i

    # Step 4: Convert triplets
    bash granularity/scripts/convert_pairs.sh

    # Step 5: 
    bash granularity/scripts/finetune.sh $i
done


