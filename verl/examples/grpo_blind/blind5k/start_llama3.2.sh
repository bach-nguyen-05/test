# export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
# CUDA_VISIBLE_DEVICES=2 python /mnt/shared/zhaonan2/envs/verl/bin/vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct     \
export HF_TOKEN=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
export HF_HOME=/mnt/shared/shared_hf_home/
CUDA_VISIBLE_DEVICES=5 python /home/asurite.ad.asu.edu/zhaonan2/envs/torchtune/bin/vllm serve meta-llama/Llama-3.2-11B-Vision-Instruct     \
    --dtype bfloat16     \
    --api-key "BLIND_VQA"    \
    --max-model-len 2048     \
    --limit-mm-per-prompt image=1     \
    --task generate     \
    --trust-remote-code     \
    --max-num-seqs 128     \
    --enforce-eager     \
    --gpu-memory-utilization 0.8    \
    --port 41651 \
    --tensor-parallel-size 1 \
    --scheduler-delay-factor 0.5 \