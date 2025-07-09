export CUDA_VISIBLE_DEVICES=1,2
export CUDA_LAUNCH_BLOCKING=1
export TMPDIR=/mnt/shared/zhaonan2/tmp  
export TEMP=/mnt/shared/zhaonan2/tmp  
export TMP=/mnt/shared/zhaonan2/tmp
export TORCH_USE_CUDA_DSA=1
export HYDRA_FULL_ERROR=1
export RAY_TEMP_DIR=/mnt/shared/zhaonan2/tmp  
export WANDB_API_KEY=e3cbbd1b589f4e74a1582314eeba28db4ba2fecd
export HF_TOKEN=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
export HF_HOME=/mnt/shared/shared_hf_home/

echo $RAY_TEMP_DIR
echo $HF_HOME

HOME=/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl

set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

blind500_train_path=$HOME/data/blind500/train.parquet
blind500_test_path=$HOME/data/blind500/test.parquet

train_files="['$blind500_train_path']"
test_files="['$blind500_test_path']"

python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager=custom \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllmConv \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    +actor_rollout_ref.rollout.start_marker="'<|im_start|>'" \
    +actor_rollout_ref.rollout.gen_start_marker="'<|im_start|>assistant'" \
    +actor_rollout_ref.rollout.end_marker="'<|im_end|>'" \
    +actor_rollout_ref.rollout.end_role_marker="''" \
    +actor_rollout_ref.rollout.max_model_len=4096 \
    +actor_rollout_ref.rollout.external_model_path=meta-llama/Llama-3.2-11B-Vision-Instruct \
    +actor_rollout_ref.rollout.external_model_port=41651 \
    +actor_rollout_ref.rollout.max_rounds=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='blind500_grpo_dev4' \
    trainer.experiment_name='qwen2_7B_conv' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=20 $@