# ===== original embedding =====

dataset=banking77
scale=small
epoch=30
d=67

OPTSTRING=":i:m:"

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

if [[$num_iteration -eq 0]]
then    
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/2_finetune/get_embedding.py \
        --model_name hkunlp/instructor-large \
        --scale $scale \
        --task_name $dataset \
        --data_path ../../datasets/${dataset}/${scale}.jsonl \
        --result_file ../../datasets/${dataset}/${scale}_embeds_iter=$1.hdf5 \
        --method $method \
        --measure
else
    checkpoint_path=perspective/2_finetune/checkpoints/finetune-pretrain-1024-gpt-noprior/instructor-large-${dataset}-d=${d}-epoch=${epoch}-iter=(($1-1)) \
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python perspective/2_finetune/get_embedding.py \
        --model_name hkunlp/instructor-large \
        --scale $scale \
        --task_name $dataset \
        --data_path ../../datasets/${dataset}/${scale}.jsonl \
        --result_file ../../datasets/${dataset}/${scale}_embeds_iter=$1.hdf5 \
        --measure \
        --checkpoint $checkpoint_path \
        --method $method \
        --overwrite
fi
 
