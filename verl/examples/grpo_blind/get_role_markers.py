import transformers
import re

# tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# def extract_conversation_from_tokens(tokenizer, token_ids, start_marker="<|im_start|>", end_marker="<|im_end|>", role_end_marker=""):
#     text = tokenizer.decode(token_ids)

#     pattern = re.escape(start_marker) + r'(.*?)' + re.escape(end_marker)
#     segments = re.findall(pattern, text, flags=re.DOTALL)
    
#     conversation = []
    
#     role_pattern = re.compile(r"^(system|assistant|user)\s+(.*)", re.IGNORECASE | re.DOTALL)
    
#     for segment in segments:
#         segment = segment.strip()
#         if not segment:
#             continue
        
#         # Use regex to match the role and content.
#         match = role_pattern.match(segment)
#         if match:
#             role = match.group(1).strip().lower()
#             content = match.group(2).strip()
            
#             # Optionally skip incomplete messages.
#             if role == "assistant":
#                 continue
            
#             conversation.append({"role": role, "content": content})
#         else:
#             # If it doesn't match, log a warning or handle it as needed.
#             raise ValueError(f"Role error: Couldn't parse segment: {segment}")
#     return conversation

def extract_conversation_from_tokens(tokenizer, token_ids, start_marker="<|im_start|>", end_marker="<|im_end|>", role_end_marker=""):
    text = tokenizer.decode(token_ids)

    pattern = re.escape(start_marker) + r'(.*?)' + re.escape(end_marker)
    segments = re.findall(pattern, text, flags=re.DOTALL)
    
    conversation = []
    
    # Adjust the role pattern based on whether a role_end_marker is provided
    if role_end_marker:
        # If there's a role end marker, match role until that marker, then content after it
        role_pattern = re.compile(
            r"^(system|assistant|user)" + 
            re.escape(role_end_marker) + 
            r"(.*)", 
            re.IGNORECASE | re.DOTALL
        )
    else:
        # Original behavior: match role followed by whitespace, then content
        role_pattern = re.compile(r"^(system|assistant|user)\s+(.*)", re.IGNORECASE | re.DOTALL)
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        
        # Use regex to match the role and content.
        match = role_pattern.match(segment)
        if match:
            role = match.group(1).strip().lower()
            content = match.group(2).strip()
            
            # Optionally skip incomplete messages.
            if role == "assistant":
                continue
            
            conversation.append({"role": role, "content": content})
        else:
            # If it doesn't match, log a warning or handle it as needed.
            raise ValueError(f"Role error: Couldn't parse segment: {segment}")
    return conversation

def format_conversation_for_model(tokenizer, conversation):
    has_assistant_last = False
    if conversation and len(conversation) > 0:
        has_assistant_last = conversation[-1]["role"] == "assistant"
    
    # Use the model's built-in chat template if available
    assert hasattr(tokenizer, "apply_chat_template")
    formatted_input = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False,
        add_generation_prompt=not has_assistant_last
    )
    token_ids = tokenizer.encode(formatted_input, return_tensors="pt")[0]
    return token_ids

test_string = "TEST_STRING"
test_messages = [
    [{"role": "system", "content": test_string}],
    [{"role": "user", "content": test_string}],
    [{"role": "assistant", "content": test_string}]
]
role_markers = {}
for role, messages in zip(["system", "user", "assistant"], test_messages):
    # Use apply_chat_template to see how each role is formatted
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    # Encode to get the token IDs for the formatted text
    formatted_ids = tokenizer.encode(formatted, add_special_tokens=True)
    # Get test content IDs to exclude from markers
    content_ids = tokenizer.encode(test_string, add_special_tokens=False)
    # Identify marker IDs by removing content IDs
    marker_ids = [id for id in formatted_ids if id not in content_ids]
    print("role marker before decode")
    print(formatted)
    print("role marker after decode")
    print(tokenizer.decode(marker_ids))
    print("-" * 20)

for msg in [
    [
        {"role": "system", "content": "my system prompt"},
        {"role": "user", "content": "my user input"}
    ]
]:
    conv_id = format_conversation_for_model(tokenizer, msg)
    conv = extract_conversation_from_tokens(tokenizer, conv_id, "<|start_header_id|>", "<|eot_id|>", "<|end_header_id|>")
    print("before ")
    print(msg)
    print("after")
    print(msg)
    print("=" * 30)