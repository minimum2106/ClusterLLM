scale=small
embed_method=finetuned
d="67.0"
dataset=banking77

# python granularity/sample_pairs.py \
#     --dataset $dataset \
#     --data_path ../../datasets/${dataset}/${scale}.jsonl \
#     --feat_path ../perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=15/checkpoint-3840/${scale}_embeds.hdf5 \
#     --scale $scale \
#     --embed_method $embed_method \
#     --k $k \
#     --out_dir granularity/sampled_pair_results \
#     --min_clusters 2 \
#     --max_clusters 200 \
#     --seed 100

if [ -f granularity/sample_pairs.py ]
then 
    echo Hello
else 
    echo Bye 
fi

# scale=large
# embed_method=finetuned
# d="67.0"
# for dataset in banking77
# do
#     for k in 1 3
#     do
#         python sample_pairs_large.py \
#             --dataset $dataset \
#             --data_path ../datasets/${dataset}/${scale}.jsonl \
#             --feat_path ../perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior-large/instructor-large-${dataset}-d=${d}-epoch=15/checkpoint-3840/${scale}_embeds.hdf5 \
#             --scale $scale \
#             --embed_method $embed_method \
#             --k $k \
#             --out_dir sampled_pair_results \
#             --min_clusters 2 \
#             --max_clusters 200 \
#             --seed 100
#     done
# done
