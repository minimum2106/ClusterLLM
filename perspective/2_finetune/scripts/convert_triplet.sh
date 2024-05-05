scale=small
dataset=banking77
python perspective/2_finetune/convert_triplet.py \
    --dataset $dataset \
    --pred_path perspective/1_predict_triplet/predicted_triplet_results/${dataset}_embed=instructor_s=${scale}_m=1024_d=67.0_sf_choice_seed=100-mistral_7b-pred.json \
    --output_path perspective/2_finetune/converted_triplet_results \
    --data_path ../../datasets/${dataset}/${scale}.jsonl
