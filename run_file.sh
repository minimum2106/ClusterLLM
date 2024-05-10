# Run the whole pipeline 
# This pipeline is to running original work of author 6 times and record the result of improved embedding model 

#!/bin/bash
num_iteration=6
epoch=30

OPTSTRING=":i:e:"

while getopts ${OPTSTRING} opt; do
  case ${opt} in
    i)
        num_iteration=${OPTARG}
        ;;

    e)
        epoch=${OPTARG}
        ;;
    
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

for method in random centroid stage2
do 
    if [[$method -eq stage2]]
    then 
        for i in $(seq 0 $num_iteration)
        do 
            # Step 1: Get embeddings from original embedding model
            # if i == 0 --> run with original embedding model 
            # if i != 0 --> run with finetuned embedding model of iteration i-1    
            bash perspective/2_finetune/scripts/get_embedding.sh $i $epoch $method

            # Step 2: Sample triplets
            bash perspective/1_predict_triplet/scripts/triplet_sampling.sh $i

            # Step 3: Predict triplets
            bash perspective/1_predict_triplet/scripts/predict_triplet.sh $i

            # Step 4: Convert triplets
            bash perspective/2_finetune/scripts/convert_triplet.sh 

            # Step 5: 
            bash perspective/2_finetune/scripts/finetune.sh -i $i -e $epoch
        done
    else
        for i in $(seq 0 $num_iteration)
        do 
            # Step 1: Get embeddings from original embedding model
            # if i == 0 --> run with the original embedding model 
            # if i != 0 --> run with the latest finetuned embedding model   
            bash perspective/2_finetune/scripts/get_embedding.sh $i $epoch $method

            # Step 2: Sample triplets
            bash granularity/scripts/sample_pairs.sh $i

            # Step 3: Predict triplets
            bash granularity/scripts/predict_pairs.sh $i

            # Step 4: Convert triplets
            bash granularity/scripts/convert_pairs.sh

            # Step 5: 
            bash granularity/scripts/finetune.sh -i $i -e $epoch
        done
    fi
done






