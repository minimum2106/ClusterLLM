iter=$1
epoch=$2
dataset=$3
seed=100

# ===== mistral-7b =====
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/2_finetune/finetune.py \
    --model_name_or_path hkunlp/instructor-large \
    --output_dir perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch}-iter=${iter} \
    --train_file perspective/2_finetune/converted_triplet_results/${dataset}_embed=instructor_s=${scale}_m=1024_choice_seed=${seed}_iter=${iter}-mistral_7b-train.json \
    --cache_dir cache \
    --max_source_length 512 \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-6 \
    --save_steps 3840 \
    --cl_temperature 0.01 \
    --overwrite_output_dir