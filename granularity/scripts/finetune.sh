epoch=30
scale=small
d="67.0"
seed=100
dataset=banking77

# ===== mistral-7b =====
train_file=granularity/predicted_pair_results/banking77_embed=finetuned_s=small_k=1_multigran2-200_seed=100-mistral_7b-prompts_pair_exps_pair_v3-train.json

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python granularity/finetune.py \
    --model_name_or_path hkunlp/instructor-large \
    --output_dir perspective/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch}-iter=$1 \
    --train_file $train_file\
    --cache_dir cache \
    --max_source_length 512 \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-6 \
    --save_steps 3840 \
    --cl_temperature 0.01 \
    --overwrite_output_dir
