# ===== original embedding =====

dataset=banking77
scale=small
epoch=30
d=67

iter=$1
epoch=$2

if [[$1 -eq 0]]
then    
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/2_finetune/get_embedding.py \
        --model_name hkunlp/instructor-large \
        --scale $scale \
        --task_name $dataset \
        --data_path ../../datasets/${dataset}/${scale}.jsonl \
        --result_file ../../datasets/${dataset}/${scale}_embeds_iter=$iter.hdf5 \
        --iter $iter \
        --epoch $epoch \
        --measure
else
    # checkpoint_path=perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch}-iter=(($1-1)) \
    checkpoint_path=perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch}-iter=$[iter - 1]
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/2_finetune/get_embedding.py \
        --model_name hkunlp/instructor-large \
        --scale $scale \
        --task_name $dataset \
        --data_path ../../datasets/${dataset}/${scale}.jsonl \
        --result_file ../../datasets/${dataset}/${scale}_embeds_iter=$iter.hdf5 \
        --measure \
        --iter $iter \
        --epoch $epoch \
        --checkpoint $checkpoint_path \
        --overwrite
fi
 
