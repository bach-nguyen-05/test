python3 scripts/model_merger.py \
    --hf_model_path Qwen/Qwen2.5-7B-Instruct \
    --local /mnt/shared/zhaonan2/checkpoints/blind500_grpo_dev4/qwen2.5_7B_conv_without_loss_mask/global_step_30/actor/ \
    --target_dir /mnt/shared/zhaonan2/checkpoints/qwen2.5_7B_hf_v4_no_loss_mask