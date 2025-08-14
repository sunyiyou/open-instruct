EVAL_LIST="manufactoria/ends_with_two_color_medium 50"
# manufactoria/basic_mix_test 27 
# manufactoria/contains_mix_test 50 
# manufactoria/numerical_comparison_test 50 
# manufactoria/regex_pattern_four_color_easy 50 
# manufactoria/regex_pattern_four_color_medium 50 
# manufactoria/regex_pattern_four_color_hard 50 
# manufactoria/regex_same_num_four_color_hard 50
# manufactoria/prepend_sequence_two_color_easy 30
# manufactoria/prepend_sequence_two_color_medium 50
# manufactoria/prepend_sequence_two_color_hard 50
# manufactoria/numerical_operations_two_color_easy 7
# manufactoria/numerical_operations_two_color_hard 14
# manufactoria/numerical_operations_two_color_medium 21
# manufactoria/numerical_max_min_two_color_medium 50
# manufactoria/manufactoria 32"

# You need to run the manufactoria server locally to run this script.

export MANUFACTORIA_API_URL=http://localhost:8071

model_name_or_path=Qwen/Qwen3-4B-Thinking-2507
exp_name=Qwen3-4B-Thinking-2507-manufactoria-all

python open_instruct/eval_code_api.py \
    --model_name_or_path $model_name_or_path \
    --api_config.model_type huggingface \
    --dataset_mixer_eval_list $EVAL_LIST \
    --dataset_mixer_eval_list_splits train \
    --dataset_transform_fn rlvr_tokenize_v1 rlvr_filter_v1 \
    --dataset_cache_mode local \
    --dataset_local_cache_dir local_dataset_cache \
    --use_last_n_eval_samples true \
    --chat_template_name qwen3 \
    --num_runs 2 \
    --seed 1 \
    --response_length 16000 \
    --temperature 1.0 \
    --vllm_top_p 1.0 \
    --vllm_num_engines 5 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9 \
    --single_gpu_mode false \
    --apply_r1_style_format_reward false \
    --apply_verifiable_reward true \
    --verification_reward 1.0 \
    --additive_format_reward false \
    --only_reward_good_outputs false \
    --non_stop_penalty true \
    --non_stop_penalty_value 0.0 \
    --manufactoria_api_url $MANUFACTORIA_API_URL/test_solution \
    --manufactoria_max_execution_time 1.0 \
    --output_dir output/eval/eval_results_${exp_name} \
    --save_results true
