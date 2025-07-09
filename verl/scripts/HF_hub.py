import os
os.environ['HF_TOKEN'] = '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'

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/mnt/shared/zhaonan2/checkpoints/qwen2_7B_hf_v4"
model_name = "qwen2_7B_rl_v4"
model_original_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_original_name)
print(model_name)
model.push_to_hub(f"zli99/{model_name}")
tokenizer.push_to_hub(f"zli99/{model_name}")
print("finished")