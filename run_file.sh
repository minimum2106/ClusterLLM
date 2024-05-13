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

for dataset in banking77 go_emotion few_rel_nat stackexchange mtop_domain
do
   for method in all_random all_centroid all_stage2
    do 
        if [[ $method = all_stage2 ]]
        then 
            for i in $(seq 0 $num_iteration)
            do 
                # Step 1: Get embeddings from original embedding model
                # if i == 0 --> run with the original embedding model 
                # if i != 0 --> run with the latest finetuned embedding model 
                echo "============================ EMBEDDINGS ============================"  
                bash perspective/2_finetune/scripts/get_embedding.sh $i $epoch $method $dataset
    
                # Step 2: Sample triplets
                echo "============================ SAMPLE ============================"
                bash granularity/scripts/sample_pairs.sh $i $dataset
    
                # Step 3: Predict triplets
                echo "============================ PREDICT ============================"
                bash granularity/scripts/predict_pairs.sh $i $dataset
    
                # Step 4: Convert triplets
                echo "============================ CONVERT ============================"
                bash granularity/scripts/convert_pairs.sh $i $dataset
    
                # Step 5: 
                echo "============================ FINETUNE ============================"
                bash granularity/scripts/finetune.sh $i $epoch $dataset
            done
        else
            for i in $(seq 0 $num_iteration)
            do 
                # Step 1: Get embeddings from original embedding model
                # if i == 0 --> run with original embedding model 
                # if i != 0 --> run with finetuned embedding model of iteration i-1    
                bash perspective/2_finetune/scripts/get_embedding.sh $i $epoch $method $dataset
                
                # Step 2: Sample triplets
                echo "============================ SAMPLE ============================"
                bash perspective/1_predict_triplet/scripts/triplet_sampling.sh $i $method $dataset
    
                # Step 3: Predict triplets
                echo "============================ PREDICT ============================"
                bash perspective/1_predict_triplet/scripts/predict_triplet.sh $i $dataset
    
                # Step 4: Convert triplets
                echo "============================ CONVERT ============================"
                bash perspective/2_finetune/scripts/convert_triplet.sh $i $dataset
    
                # Step 5: 
                echo "============================ FINETUNE ============================"
                bash perspective/2_finetune/scripts/finetune.sh $i $epoch $dataset
            done
        fi
    done
done