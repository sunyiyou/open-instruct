#!/bin/bash
set -euo pipefail
set -x

# Define datasets as a bash array (not Python syntax!)
DATASETS=(
    "manufactoria/contains_ood_mix_test" "2"
    "manufactoria/contains_mix_test" "2"
)

export MANUFACTORIA_API_URL=http://localhost:8071
model_name_or_path=Qwen/Qwen3-4B-Thinking-2507
exp_name=Qwen3-4B-Thinking-2507-manufactoria-all

# Set API key (make sure this is set in your environment)
# export ANTHROPIC_API_KEY="your-api-key-here"

cd /weka/oe-adapt-default/nouhad/open-instruct/open-instruct-fork/

PYTHONPATH_ADD="/weka/oe-adapt-default/nouhad/open-instruct/open-instruct-fork"
export PYTHONPATH="${PYTHONPATH_ADD}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
print('✅ Python working')
import open_instruct.eval_code_api as m
print('✅ Module imports successfully')
PY

# Build args safely
args=(
"open_instruct/eval_code_api.py"
--model_name_or_path "$model_name_or_path"

# API mode configuration - just set model_type, the code will handle the rest
--model_type anthropic

# Dataset configuration - use array expansion
--dataset_mixer_eval_list "${DATASETS[@]}"
--dataset_mixer_eval_list_splits train
--dataset_transform_fn rlvr_tokenize_v1 rlvr_filter_v1
--dataset_cache_mode local
--dataset_local_cache_dir local_dataset_cache
--use_last_n_eval_samples true

# Tokenizer configuration
--chat_template_name qwen3

# Evaluation configuration
--num_runs 4
--seed 1

# Generation parameters (for VLLM mode, ignored in API mode)
--response_length 16000
--temperature 1.0
--vllm_top_p 1.0
--vllm_num_engines 1
--vllm_tensor_parallel_size 1
--vllm_gpu_memory_utilization 0.9
--single_gpu_mode true

# Reward configuration
--apply_r1_style_format_reward false
--apply_verifiable_reward true
--verification_reward 1.0
--additive_format_reward false
--only_reward_good_outputs false
--non_stop_penalty true
--non_stop_penalty_value 0.0

# Tool configuration
--manufactoria_api_url "$MANUFACTORIA_API_URL/test_solution"
--manufactoria_max_execution_time 1.0

# Output configuration
--output_dir "output/eval/eval_results_${exp_name}"
--save_results true
)

python "${args[@]}"