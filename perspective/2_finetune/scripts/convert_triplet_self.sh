scale=small
dataset=banking77
python convert_triplet_self.py \
    --dataset $dataset \
    --pred_path ../1_predict_triplet/predicted_triplet_results/${dataset}_embed=instructor_s=${scale}_m=1024_d=67.0_sf_choice_seed=100-mistral_7b-pred.json \
    --output_path converted_triplet_results \
    --feat_path ../../../../datasets/${dataset}/${scale}_embeds_iter=$1.hdf5 \
    --data_path ../../../../datasets/${dataset}/${scale}.jsonl

# --feat_path ../../../../datasets/${dataset}/${scale}_embeds.hdf5 \