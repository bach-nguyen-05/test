from shared.prompt import txt_system_prompt, vsl_system_prompt, txt_cot_system_prompt, e2e_system_prompt
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def initialize_msg(question, image_path, encode_images=False, cot=False):
    system_prompt = txt_system_prompt
    if cot:
        system_prompt = txt_cot_system_prompt
    txt_msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    if encode_images:
        base64_image = encode_image(image_path)
        img_type = image_path.split(".")[-1]
        vsl_msg = [
            {"role": "system", "content": vsl_system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{img_type};base64,{base64_image}"}
                    }
                ]
            }
        ]
        return txt_msg, vsl_msg
    else:
        vsl_msg = [
            {"role": "user", "content": vsl_system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path}
                    }
                ]
            }
        ]
        return txt_msg, vsl_msg

def initialize_e2e_msg(question, image_path, encode_images=False):
    vsl_system_prompt = e2e_system_prompt
    if encode_images:
        base64_image = encode_image(image_path)
        img_type = image_path.split(".")[-1]
        vsl_msg = [
            {"role": "system", "content": vsl_system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{img_type};base64,{base64_image}"}
                    }
                ]
            },
            {"role": "user", "content": question}
        ]
        return vsl_msg
    else:
        vsl_msg = [
            {"role": "user", "content": vsl_system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_path}
                    }
                ]
            },
            {"role": "user", "content": question}
        ]
    return vsl_msg

import copy, re

# Check if the response contains an answer.
def is_answer_found(text):
    return "the answer is" in text.strip().lower()

# match anything after "My question is" or "The answer is"
def remove_possible_cot_old(text):
    import re
    pattern = re.compile(
        r'(?i)\b(?:the answer is|my question is)\b.*',  
        re.IGNORECASE | re.DOTALL                       
    )
    match = pattern.search(text)
    if match:
        return match.group(0).strip().rstrip("\"").strip()
    # Fallback
    return "There's no provided question. Please ignore this message."

def extract_after_answer_old(answer):
    pattern = re.compile(
        r'\bthe answer is\b\s*(.*)',  # marker + any whitespace + capture the rest
        re.IGNORECASE | re.DOTALL      # case‑insensitive, dot matches newlines
    )
    match = pattern.search(answer)
    if match:
        return match.group(1).strip().lstrip(":").rstrip("\"").rstrip(".").strip()
    return ""

def remove_possible_cot(text: str) -> str:
    """
    Return the *trailing* chunk that starts with the final occurence of either
    'the answer is' or 'my question is'.  If neither appears, fall back.
    """
    # compile once, case-insensitive, spanning multiple lines
    marker_re = re.compile(r'\b(?:the answer is|my question is)\b', re.IGNORECASE)

    matches = list(marker_re.finditer(text))
    if not matches:
        return "There's no provided question. Please ignore this message."

    # Position of the last marker
    last = matches[-1]
    tail = text[last.start():]          # keep marker + everything after
    return tail.strip().rstrip('"').strip()


def extract_after_answer(answer: str) -> str:
    """
    Extract whatever follows the *final* 'the answer is' (colon optional),
    stripping trailing quotes / periods / whitespace.
    """
    marker_re = re.compile(r'\bthe answer is\b\s*', re.IGNORECASE)

    matches = list(marker_re.finditer(answer))
    if not matches:
        return ""                       # nothing to extract

    last = matches[-1]
    # Slice *after* the marker
    raw = answer[last.end():]

    # clean up leading punctuation and trailing adornments
    cleaned = raw.lstrip(":").strip()
    cleaned = cleaned.rstrip('"').rstrip(".").strip()
    return cleaned

def extract_mcq_option(answer: str):
    """
    Extracts a single MCQ option like (A), (B), (C), or (D) from the answer string.
    Returns the option with parentheses if exactly one is found; otherwise returns None.
    """
    # Find all occurrences of (A), (B), (C), or (D)
    matches = re.findall(r'\(([A-D])\)', answer)
    
    if len(matches) == 1:
        # Re‑wrap the single letter in parentheses and return
        return f"({matches[0]})"
    # Zero or multiple matches → reject
    return ""

if __name__ == "__main__":
    # Test the functions
    test_cases = [
        "The answer is (A) 42.",
        "Thought: I think the answer is (A) Weight: [(A), (B), (C)] Action: The answer is (A).",
        "Thought: I think the answer is (A), but the answer is (B) Weight: [(A), (B), (C)] Action: The answer is (A).",
        "my question is: What is the capital of France? The answer is (B) Paris.",
        "my question is: What is the capital of France?",
        "some thinking process my question is: What is the capital of France?",
        "some random text",
    ]

    for test in test_cases:
        print(f"Test case: {test}")
        print(f"Extracted answer:\n {extract_after_answer(remove_possible_cot(test))}")
        print(f"Extracted answer new:\n {extract_after_answer_new(remove_possible_cot_new(test))}")

        print(f"Removed possible COT:\n {remove_possible_cot(test)}")
        print(f"Removed possible COT new:\n {remove_possible_cot_new(test)}")

        print("-" * 50)