description="manufactoria_basic_test"
exp_name=Qwen3_4B_Instruct_manufactoria_basic_test
    
python mason.py \
    --cluster ai2/jupiter-cirrascale-2 \
    --workspace ai2/tulu-thinker \
    --priority high \
    --preemptible \
    --num_nodes 2 \
    --image yiyous/cuda12.8-dev-ubuntu22.04-torch2.7.0 \
    --description "${description}" \
    --budget ai2/oe-adapt \
    --gpus 8 -- source configs/beaker_configs/ray_node_setup.sh \&\&  \
    source configs/beaker_configs/manufactoria_api_setup.sh \&\& \
    export BEAKER_TOKEN= \&\& \
    export WANDB_API_KEY= \&\& \
    export HF_TOKEN= \&\& \
    python open_instruct/grpo_fast_code.py \
    --exp_name $exp_name \
    --beta 0.01 \
    --num_unique_prompts_rollout 48 \
    --num_samples_per_prompt_rollout 16 \
    --try_launch_beaker_eval_jobs_on_weka \
    --kl_estimator kl3 \
    --learning_rate 5e-7 \
    --dataset_mixer_list manufactoria/basic_mix_train 1.0\
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list manufactoria/basic_mix_test 27 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 10240 \
    --max_prompt_token_length 2048 \
    --response_length 8192 \
    --pack_length 10240 \
    --model_name_or_path Qwen/Qwen3-4B-Instruct-2507 \
    --apply_verifiable_reward true \
    --manufactoria_api_url \$MANUFACTORIA_API_URL/test_solution \
    --non_stop_penalty True \
    --non_stop_penalty_value 0.0 \
    --temperature 1.0 \
    --chat_template_name qwen3 \
    --total_episodes 1000000 \
    --deepspeed_stage 2 \
    --per_device_train_batch_size 1 \
    --num_mini_batches 2 \
    --num_learners_per_node 8 \
    --num_epochs 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_num_engines 8 \
    --lr_scheduler_type constant \
    --seed 1 \
    --num_evals 100 \
    --save_freq 40 \
    --gradient_checkpointing \
    --with_tracking