# ===== original embedding =====

dataset=banking77
scale=small
epoch=30
d=67

# if [[$1 -eq 0]]
# then    
#     CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
#         --model_name hkunlp/instructor-large \
#         --scale $scale \
#         --task_name $dataset \
#         --data_path ../../../../datasets/${dataset}/${scale}.jsonl \
#         --result_file ../../../../datasets/${dataset}/${scale}_embeds_iter=$1.hdf5 \
#         --measure
# else
#     checkpoint_path=checkpoints/finetune-pretrain-1024-gpt-noprior/'instructor-large-banking77-d=67.0-epoch=15'/checkpoint-3840-iter=$1
#     checkpoint_path=checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch}-iter=$1 \
#     CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
#         --model_name hkunlp/instructor-large \
#         --scale $scale \
#         --task_name $dataset \
#         --data_path ../../../../datasets/${dataset}/${scale}.jsonl \
#         --result_file ../../../../datasets/${dataset}/${scale}_embeds_iter=$1.hdf5 \
#         --measure \
#         --checkpoint $checkpoint_path \
#         --overwrite
# fi

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
    --model_name hkunlp/instructor-large \
    --scale $scale \
    --task_name $dataset \
    --data_path ../../../../datasets/${dataset}/${scale}.jsonl \
    --result_file ../../../../datasets/${dataset}/${scale}_embeds_iter=$1.hdf5 \
    --measure
 