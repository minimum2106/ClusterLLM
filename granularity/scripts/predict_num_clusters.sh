# ----- small -----
# scale=small
# # --num_data 1024 \
# for dataset in 
# do
#     for embed in finetuned
#     do
#         for k in 1
#         do
#             python predict_num_clusters.py \
#                 --dataset $dataset \
#                 --embed_method $embed \
#                 --clustering_results sampled_pair_results/${dataset}_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json \
#                 --pred_path predicted_pair_results/${dataset}_embed=finetuned_s=small_k=1_multigran2-200_seed=100-gpt-4-0314-prompts_pair_exps_pair_v3.json \
#                 --min_clusters 2 \
#                 --max_clusters 200
#         done
#     done
# done

# ----- large -----
scale=small
model=mistral_7b
for dataset in banking77
do
    for embed in finetuned
    do
        for k in 1
        do
            python predict_num_clusters.py \
                --dataset $dataset \
                --embed_method $embed \
                --scale ${scale} \
                --data_path ../../../unlabeled_data/${dataset}/${scale}.jsonl \
                --clustering_results sampled_pair_results/${dataset}_embed=finetuned_s=${scale}_k=${k}_multigran2-200_seed=100.json \
                --pred_path predicted_pair_results/${dataset}_embed=finetuned_s=${scale}_k=${k}_multigran2-200_seed=100-${model}-prompts_pair_exps_pair_v3.json \
                --min_clusters 2 \
                --max_clusters 200
        done
    done
done