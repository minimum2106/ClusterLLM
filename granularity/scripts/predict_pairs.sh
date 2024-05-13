dataset=$2

link_path=granularity/sampled_pair_results/${dataset}_embed=finetuned_s=small_k=1_multigran2-200_seed=100.json
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python granularity/predict_pairs.py \
    --dataset $dataset \
    --data_path $link_path \
    --model_name mistral_7b \
    --openai_org "OPENAI_ORG" \
    --prompt_file granularity/prompts_pair_exps_pair_v3.json \
    --temperature 0