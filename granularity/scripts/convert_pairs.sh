scale=small
dataset=$2
pred_path=granularity/predicted_pair_results/banking77_embed=finetuned_s=small_k=1_multigran2-200_seed=100-mistral_7b-prompts_pair_exps_pair_v3-pred.json

python granularity/convert_pairs.py \
    --dataset $dataset \
    --pred_path $pred_path \
    --output_path granularity/converted_pair_results \
    --data_path ../../datasets/${dataset}/${scale}.jsonl


