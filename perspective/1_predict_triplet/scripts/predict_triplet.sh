dataset=$2
link_path=perspective/1_predict_triplet/sampled_triplet_results/${dataset}_embed=instructor_s=small_m=1024_choice_seed=100_iter=$1.json

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/1_predict_triplet/predict_triplet.py \
    --dataset $dataset \
    --data_path $link_path \
    --openai_org "OPENAI_ORG" \
    --model_name mistral_7b \
    --temperature 0





