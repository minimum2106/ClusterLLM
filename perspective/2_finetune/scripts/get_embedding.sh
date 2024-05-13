# ===== original embedding =====

scale=small
d=67

iter=$1
epoch=$2
method=$3
dataset=$4


if [[ $iter -eq 0 ]]
then    
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/2_finetune/get_embedding.py \
        --model_name hkunlp/instructor-large \
        --scale $scale \
        --task_name $dataset \
        --data_path ../../datasets/${dataset}/${scale}.jsonl \
        --result_file ../../datasets/${dataset}/${scale}_embeds_iter=$iter.hdf5 \
        --iter $iter \
        --epoch $epoch \
        --method $method \
        --measure \
        --overwrite
else
    checkpoint_path=perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=-epoch=${epoch}-iter=$[iter - 1] 
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/2_finetune/get_embedding.py \
        --model_name hkunlp/instructor-large \
        --scale $scale \
        --task_name $dataset \
        --data_path ../../datasets/${dataset}/${scale}.jsonl \
        --result_file ../../datasets/${dataset}/${scale}_embeds_iter=$iter.hdf5 \
        --measure \
        --checkpoint $checkpoint_path \
        --iter $iter \
        --epoch $epoch \
        --method $method \
        --overwrite
fi