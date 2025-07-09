import re
import json
from shared.utils import remove_possible_cot, extract_after_answer, extract_mcq_option
import sys
# temporarily remove current directory from module search path
cwd = sys.path.pop(0)    
import math    # now this is the real std-lib math
# put things back how you found them
sys.path.insert(0, cwd)

def extract_weight_json(text):
    pattern = r"""
    \[\s*                         # Opening bracket of the array
    (?:\{\s*                      # Opening of the first dict
    (?:"[^"]+"\s*:\s*(?:0(?:\.\d+)?|1(?:\.0+)?))  # First key-value pair (float between 0 and 1)
    (?:\s*,\s*"[^"]+"\s*:\s*(?:0(?:\.\d+)?|1(?:\.0+)?))*  # More key-value pairs
    \s*\}\s*,?\s*)+              # Closing brace, optional comma
    \]                           # Closing bracket of the array
    """
    # Compile with re.VERBOSE so we can use comments
    compiled = re.compile(pattern, re.VERBOSE)
    match = compiled.search(text)
    if match:
        return match.group()
    return None

def format_reward(conv) -> float:
    for msg in conv:
        if msg["role"] == "assistant":
            content = remove_possible_cot(msg["content"]).lower()
            if "the answer is" not in content and "my question is" not in content:
                return 0.0
            if "the answer is" in content and "my question is" in content:
                return 0.0
    final_prediction = extract_final_prediction(conv)
    possible_answer = extract_after_answer(final_prediction)
    possible_mcq_option = extract_mcq_option(possible_answer)
    if possible_mcq_option == "":
        return 0.0
    return 1.0

def format_reward_new_scheme(conv, breakdown=False):
    marker_score = 0.5 # has answer or question
    weight_score = 0.5 # has parsable json
    # check for question or final answer, as usual
    for msg in conv:
        if msg["role"] == "assistant":
            content = remove_possible_cot(msg["content"]).lower()
            if "the answer is" not in content and "my question is" not in content:
                marker_score = 0.0
                break
            if "the answer is" in content and "my question is" in content:
                marker_score = 0.0
                break

    final_prediction = extract_final_prediction(conv)
    possible_answer = extract_after_answer(final_prediction)
    possible_mcq_option = extract_mcq_option(possible_answer)
    if possible_mcq_option == "":
        marker_score = 0.0

    # check for parsable json (they are weight assignment)
    last_conf_prob = None
    for msg in conv:
        if msg["role"] == "assistant":
            content = msg["content"]
            weight = extract_weight_json(content)
            if weight is None:
                # print("no parsable json")
                weight_score = 0.0
                break
            try:
                data = json.loads(weight)
                if type(data) != list:
                    # print("json not correct format")
                    weight_score = 0.0
                    break
                conf = data[0]
                conf_prob = [conf[k] for k in conf]
                if last_conf_prob is not None:
                    # length check
                    if len(last_conf_prob) != len(conf_prob):
                        weight_score = 0.0
                        # print("length change")
                        break
                last_conf_prob = conf_prob
                # print("extracted prb", conf_prob)
                if min(conf_prob) < 0 or max(conf_prob) > 1 or sum(conf_prob) != 1.0:
                    weight_score = 0.0
                    # print("not prob. dist.")
                    break
            except Exception as e:
                # print("unknow error", e)
                weight_score = 0.0
                break
    if last_conf_prob is not None and weight_score > 0 and marker_score > 0:
        if max(last_conf_prob) < 0.7:
            weight_score = 0.0
            # weight_score = min(weight_score, 0.25) # give partial reward if only last prob. is not correct
    else:
        weight_score = 0.0

    total_score = marker_score + weight_score
    if breakdown:
        return total_score, {
            "marker": marker_score,
            "weight": weight_score
        }
    return total_score

def acc_reward(predict_str: str, ground_truth: str) -> float:    
    answer = extract_after_answer(predict_str)
    # print("answer", answer, "ground truth", ground_truth.strip())
    if answer == ground_truth.strip() or extract_mcq_option(answer) == ground_truth.strip():
        return 1.0
    else:
        return 0.0

def extract_final_prediction(conv):
    assert conv[-1]["role"] == "assistant"
    ans = conv[-1]["content"]
    return ans

def get_conv_round(conv):
    num = 0
    for msg in conv:
        if msg["role"] == "assistant":
            num += 1
    return max(0, num - 1)

def length_reward(n):
    return 1 - math.exp(-n / 2)

def compute_score(conv, ground_truth, new_scheme=False, breakdown=False) -> float:
    final_prediction = extract_final_prediction(conv)
    max_fmt_score = 0.1
    if not new_scheme:
        fmt_reward_score = 0.1 * format_reward(conv)
        acc_reward_score = 1.0 * acc_reward(final_prediction, ground_truth) if max_fmt_score != fmt_reward_score else 0.0
        len_reward_score = 0.0 * length_reward(get_conv_round(conv)) if max_fmt_score != fmt_reward_score else 0.0

        total_score = acc_reward_score + fmt_reward_score + len_reward_score
        if breakdown:
            components = {
                "acc": acc_reward_score, 
                "fmt": fmt_reward_score, 
                "len": len_reward_score
            }
            components["total"] = total_score
            components["conv_round"] = get_conv_round(conv)
            return total_score, components
        return total_score
    else:
        fmt_reward_score = 0.1 * format_reward_new_scheme(conv)
        acc_reward_score = 1.0 * acc_reward(final_prediction, ground_truth) if max_fmt_score == fmt_reward_score else 0.0
        len_reward_score = 0.0 * length_reward(get_conv_round(conv)) if max_fmt_score == fmt_reward_score else 0.0
        
        total_score = acc_reward_score + fmt_reward_score + len_reward_score
        if breakdown:
            components = {
                "acc": acc_reward_score, 
                "fmt": fmt_reward_score, 
                "len": len_reward_score
            }
            components.update(format_reward_new_scheme(conv, breakdown=True)[1])
            components["total"] = total_score
            components["conv_round"] = get_conv_round(conv)
            
            # log the scores
            # with open("/home/asurite.ad.asu.edu/zhaonan2/blind_project/verl/_debug_output/blind_scores_new.log", "a") as f:
            #     for k, v in components.items():
            #         f.write(f"{k}: {v},")
            #     f.write("\n")

            return total_score, components
        return total_score

if __name__ == "__main__":
    # ans = [
    #     ["The answer is (A)", "(A)"],
    #     ["The answer is (A).", "(A)"],
    #     ["The answer is: (A)", "(A)"],
    #     ["The answer is: (A).", "(A)"],
    #     ["The answer is: (A) yes.", "(A)"],
    #     ["The answer is (A) yes.", "(A)"],
    #     ["Some random string", "(A)"]
    # ]
    ans = [
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "The answer is (A)"},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "my cot \n The answer is (A)."},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "my cot \n my fist question"},
            {"role": "user", "content": "my first answer"},
            {"role": "assistant", "content": "my cot \n The answer is (A)."},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "my cot \n My question is my fist question"},
            {"role": "user", "content": "my first answer"},
            {"role": "assistant", "content": "my cot but not separated by new line. The answer is (A)."},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "my cot \n my another cot \n The answer is: (A) yes."},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "my cot \n The answer is (A) yes."},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "I guess the answer is (A) and (B)."},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "Some random string"},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "[{\"A\": 0.7, \"B\": 0.1, \"C\": 0.2}] the answer is (A)"},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content": "[{\"A\": 0.5, \"B\": 0.1, \"C\": 0.4}] my question is something"},
            {"role": "user", "content": "my second question"},
            {"role": "assistant", "content": "[{\"A\": 0.7, \"B\": 0.3}] the answer is (B)"},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content":  "[{\"A\": 0.3, \"B\": 0.7] the answer is (A)"},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content":  "[{\"A\": 0.5, \"B\": 0.5}] my question is some question"},
            {"role": "user", "content": "answer 1"},
            {"role": "assistant", "content":  "[{\"A\": 0.5, \"B\": 0.5}] my question is some question"},
            {"role": "user", "content": "answer 1"},
            {"role": "assistant", "content":  "[{\"A\": 0.7, \"B\": 0.3}] the answer is maybe not (B), but (A)"},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content":  "[{\"A\": 0.5, \"B\": 0.5}] my question is some question"},
            {"role": "user", "content": "answer 1"},
            {"role": "assistant", "content":  "[{\"A\": 0.5, \"B\": 0.5}] my question is some question"},
            {"role": "user", "content": "answer 1"},
            {"role": "assistant", "content":  "[{\"A\": 0.7, \"B\": 0.3}] the answer is maybe (A)"},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content":  "[{\"A\": 0.5, \"B\": 0.5}] my question is some question"},
            {"role": "user", "content": "answer 1"},
            {"role": "assistant", "content":  "[{\"A\": 0.5, \"B\": 0.5}] my question is some question"},
            {"role": "user", "content": "answer 1"},
            {"role": "assistant", "content":  "[{\"A\": 0.7, \"B\": 0.3}] the answer is maybe not (B)"},
        ],
        [
            {"role": "user", "content": "my first question"},
            {"role": "assistant", "content":  "[{\"A\": 0.5, \"B\": 0.5}] my question is some question"},
            {"role": "user", "content": "answer 1"},
            {"role": "assistant", "content":  "[{\"A\": 0.5, \"B\": 0.5}] my question is some question"},
            {"role": "user", "content": "answer 2"},
            {"role": "assistant", "content":  "[{\"A\": 0.7, \"B\": 0.3}] my question is some question"},
            {"role": "user", "content": "answer 3"},
            {"role": "assistant", "content":  "[{\"A\": 0.8, \"B\": 0.2}] the answer is maybe (A)"},
        ]
    ]


    for pred in ans:
        label = "(A)"
        print("prediction: ", pred, "label", label)
        print(compute_score(pred, label, True, True))
        print()