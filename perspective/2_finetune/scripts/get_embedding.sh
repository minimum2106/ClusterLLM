# ===== original embedding =====
# for dataset in banking77 few_rel_nat stackexchange go_emotion
# do
#     for scale in small
#     do
#         CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
#             --model_name hkunlp/instructor-large \
#             --scale $scale \
#             --task_name $dataset \
#             --data_path ../../../../datasets/${dataset}/${scale}.jsonl \
#             --result_file ../../../../datasets/${dataset}/${scale}_embeds.hdf5 \
#             --measure
#     done
# done

# ===== with checkpoint =====
# scale=small
# 
for dataset in banking77
do
    for scale in small
    do
        checkpoint_path=checkpoints/finetune-pretrain-1024-gpt-noprior/'instructor-large-banking77-d=67.0-epoch=15'/checkpoint-3840
        CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding.py \
            --model_name hkunlp/instructor-large \
            --scale $scale \
            --task_name $dataset \
            --data_path ../../../../datasets/${dataset}/${scale}.jsonl \
            --result_file ${checkpoint_path}/${scale}_embeds.hdf5 \
            --measure \
            --checkpoint $checkpoint_path \
            --overwrite
    done 
done
