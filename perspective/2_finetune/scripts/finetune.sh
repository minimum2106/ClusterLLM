scale=small
d="67.0"
seed=100
dataset=banking77

OPTSTRING=":i:e:"
while getopts ${OPTSTRING} opt; do
  case ${opt} in
    i)
        num_iteration=${OPTARG}
        echo helo
        ;;

    e)
        epoch=${OPTARG}
        ;;
    
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# ===== mistral-7b =====
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/2_finetune/finetune.py \
    --model_name_or_path hkunlp/instructor-large \
    --output_dir perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch}-iter=$num_iteration \
    --train_file perspective/2_finetune/converted_triplet_results/${dataset}_embed=instructor_s=${scale}_m=1024_d=${d}_sf_choice_seed=${seed}-mistral_7b-train.json \
    --cache_dir cache \
    --max_source_length 512 \
    --num_train_epochs $epoch \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-6 \
    --save_steps 3840 \
    --cl_temperature 0.01 \
    --overwrite_output_dir
