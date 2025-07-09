import pandas as pd
import transformers

model_name = "Qwen/Qwen2.5-7B-Instruct"
model_path = "/mnt/shared/zhaonan2/checkpoints/qwen2_7B_hf"
data_path = "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl/data/blind500/test.parquet"

data = pd.read_parquet(data_path)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
# model = transformers.AutoModel.from_pretrained(model_path)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path)

for i in range(20):
    msg = data.iloc[i]["prompt"]
    # print("msg")
    # print(msg)
    # print("*" * 20)
    input_text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    input_text = tokenizer(
        input_text,
        add_special_tokens=False,
        return_tensors="pt")
    # print("input_text")
    # print(input_text)
    # print("*" * 20)
    output = model.generate(**input_text, temperature=0, max_new_tokens=128)
    output = tokenizer.decode(output[0], skip_special_tokens=False)
    print("output")
    print(output)
    print("*" * 20)
    print("ground truth")
    print(data.iloc[i]["extra_info"]["answer"])
    print("*" * 20)
