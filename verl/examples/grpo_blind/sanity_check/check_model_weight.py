import transformers
import os
import torch

# add env variable
os.environ["HF_HOME"] = "/mnt/shared/shared_hf_home/"
os.environ["HF_TOKEN"] = "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"

model_name_1 = "Qwen/Qwen2.5-7B-Instruct"
# model_name_2 = "zli99/qwen2.5_7B_rl_v4"
model_name_2 = "/mnt/shared/zhaonan2/checkpoints/qwen2.5_7B_hf_v4_no_loss_mask"
tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# load the model
model_1 = transformers.AutoModelForCausalLM.from_pretrained(model_name_1)
model_2 = transformers.AutoModelForCausalLM.from_pretrained(model_name_2)

flag = False
# compare the model weights
for name, param_1 in model_1.named_parameters():
    param_2 = model_2.get_parameter(name)
    if not torch.equal(param_1, param_2):
        print(f"Model weights differ at {name}")
        flag = True

if not flag:
    print("all model weights are the same")

TXT_SYSTEM_PROMPT = '''You are a visually impaired person. Imagine there's an image, and you are asked to answer 
a multiple-choice question about it. There's a visual interpreter you can talk to, but it can only answer 
perception-oriented queries, such as shapes, colors, textures, and spatial relationships. You are tasked to collect 
these visual details and reason about them to derive your final answer. Sometimes, the visual interpreter's answer 
can be incomplete or inaccurate, so you need to employ the process of elimination and make sure your final answer 
outweighs all other options. There's an upper limit to the number of questions you can ask, so start with broad, 
decisive questions that can rapidly narrow down the possibilities. In your conversation with the visual interpreter, 
you ask only one question at a time, and you are NOT allowed to disclose the original question in any way, 
including repeating or paraphrasing it. Only after you gather enough information to derive your final answer to the 
original question should you output "The answer is:" followed by just the letter of the correct option ((A), (B), 
(C), (D)), without any additional explanation. You do not need to reveal your reasoning process. If your question 
gets rejected by the visual interpreter, you do not need to apologize; try to break it down into more straightforward 
questions or ask from a different perspective. Now, let's ask the first question for the following question.'''.replace("\n", " ")

msg = [{
    "role": "system",
    "content": TXT_SYSTEM_PROMPT
},{
    "role": "user",
    "content": "Have a lot of people skied through this location?\nSelect from the following choices:\n(A) yes\n(B) no"
},{
    "role": "assistant",
    "content": "What is the general activity seen in the image?"
},{
    "role": "user",
    "content": "There are some people skiing."
}]

# generate the response from model_1 and model_2


# encode the message by first applying chat template
msg = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
# append <|im_start|>user to the input_ids
# msg = msg + " <|im_start|>user"
# encode the message
inputs = tokenizer(msg, return_tensors="pt")

# generate the response
response1 = model_1.generate(**inputs)
response2 = model_2.generate(**inputs)
# decode the response
response1 = tokenizer.batch_decode(response1, skip_special_tokens=False)[0]
response2 = tokenizer.batch_decode(response2, skip_special_tokens=False)[0]
# print the response
print("model_1 response: \n", response1)
print("=" * 20)
print("model_2 response: \n", response2)