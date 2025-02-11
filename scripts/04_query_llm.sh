#!/bin/bash
set -e  # Exit on error

# Set environment variables
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Directories and configurations
dataset_dir="C:/Users/ankit/Documents/LLM/lavad/datasets"
llm_model_name="llama-2-7b-chat"
batch_size=1
frame_interval=16
#index_name="opt-6.7b-coco+opt-6.7b+flan-t5-xxl+flan-t5-xl+flan-t5-xl-coco"
index_name="blip2-flan-t5-xl-coco"

echo "Processing index: $index_name"

# Paths
root_path="${dataset_dir}/frames"
annotationfile_path="${dataset_dir}/annotations/test.txt"
captions_dir="$dataset_dir/captions/clean/$index_name/"

# Prompts
context_prompt="If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
format_prompt="Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'"
summary_prompt="Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."

# Generate unique experiment ID and directory name
exp_id=$(date +%s | tail -c 7)
dir_name=$(echo "$context_prompt" | tr '[:upper:]' '[:lower:]' | tr ' ' '_' | cut -c1-243)
dir_name=$(printf "%s_%s" "$exp_id" "$dir_name")

output_scores_dir="${dataset_dir}/scores/raw/${llm_model_name}/${index_name}/${dir_name}/"
output_summary_dir="${dataset_dir}/captions/summary/${llm_model_name}/${index_name}/"

# Run the summary generation
torchrun --nproc_per_node=1 --nnodes=1 --rdzv_backend=static --rdzv_endpoint=localhost:29500 -m src.models.llm_anomaly_scorer \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --summary_prompt "$summary_prompt" \
    --output_summary_dir "$output_summary_dir" \
    --captions_dir "$captions_dir" \
    --ckpt_dir libs/llama/llama-2-7b-chat/ \
    --tokenizer_path libs/llama/llama-2-7b-chat/tokenizer.model

# Run the scoring generation
torchrun --nproc_per_node=1 --nnodes=1 --rdzv_backend=static --rdzv_endpoint=localhost:29500 -m src.models.llm_anomaly_scorer \
    --root_path "$root_path" \
    --annotationfile_path "$annotationfile_path" \
    --batch_size "$batch_size" \
    --frame_interval "$frame_interval" \
    --output_summary_dir "$output_summary_dir" \
    --context_prompt "$context_prompt" \
    --format_prompt "$format_prompt" \
    --output_scores_dir "$output_scores_dir" \
    --ckpt_dir libs/llama/llama-2-7b-chat/ \
    --tokenizer_path libs/llama/llama-2-7b-chat/tokenizer.model \
    --score_summary
