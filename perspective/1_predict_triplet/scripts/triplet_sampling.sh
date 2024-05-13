# ===== instructor-large =====
scale=small
max_query=1024
embed=instructor

iter=$1
method=$2
dataset=$3

feat_path=../../datasets/${dataset}/${scale}_embeds_iter=${iter}.hdf5 perspective/1_predict_triplet/scripts/triplet_sampling.sh
python perspective/1_predict_triplet/triplet_sampling.py \
    --data_path ../../datasets/${dataset}/${scale}.jsonl \
    --feat_path $feat_path \
    --dataset $dataset \
    --embed_method $embed \
    --max_query $max_query \
    --filter_first_prop 0.0 \
    --large_ent_prop 0.2 \
    --out_dir perspective/1_predict_triplet/sampled_triplet_results \
    --max_distance 67 \
    --scale $scale \
    --shuffle_inds \
    --seed 100 \
    --iter $1 \
    --method $method \


# for dataset in banking77 
# do
#     for max_query in 1024
#     do
#         for embed in instructor
#         do
#             feat_path=../../../../datasets/${dataset}/${scale}_embeds.hdf5
#             python triplet_sampling.py \
#                 --data_path ../../../../datasets/${dataset}/${scale}.jsonl \
#                 --feat_path $feat_path \
#                 --dataset $dataset \
#                 --embed_method $embed \
#                 --max_query $max_query \
#                 --filter_first_prop 0.0 \
#                 --large_ent_prop 0.2 \
#                 --out_dir sampled_triplet_results \
#                 --max_distance 67 \
#                 --scale $scale \
#                 --shuffle_inds \
#                 --seed 100
#         done
#     done
# done

# ===== e5-large =====
# scale=small
# for dataset in banking77 few_rel_nat stackexchange go_emotion
# do
#     for max_query in 1024
#     do
#         for embed in e5
#         do
#             feat_path=../../datasets/${dataset}/${scale}_embeds_e5.hdf5
#             python triplet_sampling.py \
#                 --data_path ../../datasets/${dataset}/${scale}.jsonl \
#                 --feat_path $feat_path \
#                 --dataset $dataset \
#                 --embed_method $embed \
#                 --max_query $max_query \
#                 --filter_first_prop 0.0 \
#                 --large_ent_prop 0.2 \
#                 --out_dir sampled_triplet_results \
#                 --max_distance 77 \
#                 --scale $scale \
#                 --shuffle_inds \
#                 --seed 100
#         done
#     done
# done

# ===== finetuned embedding =====
# scale=small
# for dataset in banking77 
# do
#     for max_query in 1024
#     do
#         for embed in finetuned
#         do
#             feat_path=../../2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-epoch=15/checkpoint-3840/${scale}_embeds.hdf5
#             python triplet_sampling.py \
#                 --data_path ../../../../datasets/${dataset}/${scale}.jsonl \
#                 --feat_path $feat_path \
#                 --dataset $dataset \
#                 --embed_method $embed \
#                 --max_query $max_query \
#                 --filter_first_prop 0.0 \
#                 --large_ent_prop 0.2 \
#                 --out_dir sampled_triplet_results \
#                 --max_distance 67 \
#                 --scale $scale \
#                 --shuffle_inds \
#                 --seed 100
#         done
#     done
# done