import json
from evaluation.utils import *
from shared.utils import *

predict_file = "/home/asurite.ad.asu.edu/zhaonan2/blind_project/results/qwen2.5_7B_original_spubench_500_sc=True.json"
test_set = "spubench_500"

# ground_truth_file = "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/blind-vlm-project/datasets/eval/v5_val_724.json"
# img_dir = "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/blind-vlm-project/datasets/eval/v5_eval"

# ground_truth_file = "/mnt/shared/shijie/blind-vlm-project/delete_me/perturbed_613.json"
# img_dir = "/mnt/shared/shijie/blind-vlm-project/delete_me/"

# ground_truth_file = "/home/asurite.ad.asu.edu/zhaonan2/blind_project/verl/data/blind5k_new_reasoning_scheme/test_mcq_v2.json"

test_sets = {
    "724": {
        "test_file": "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/blind-vlm-project/datasets/eval/v5_val_724.json",
        "img_dir": "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/blind-vlm-project/datasets/eval/v5_eval"
    },
    "613": {
        "test_file": "/mnt/shared/shijie/blind-vlm-project/delete_me/perturbed_613.json",
        "img_dir": "/mnt/shared/shijie/blind-vlm-project/delete_me/"
    },
    "val2": {
        "test_file": "/home/asurite.ad.asu.edu/zhaonan2/blind_project/verl/data/blind5k_new_reasoning_scheme/test_mcq_v2.json",
        "img_dir": ""
    },
    "spubench_500": {
        "test_file": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/MM-SpuBench/data/mmspubench/data.json",
        "img_dir": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/MM-SpuBench/data/mmspubench"
    },
    "CounterAnimal": {
        "test_file": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/CounterAnimal/data.json",
        "img_dir": "/mnt/shared/shijie/blind-vlm-project/new3_dataset/CounterAnimal/"
    }
}

# prediction_dict = {}
# for run in range(runs):
#     name = f"run {run}"
#     dataset_preds = [] # a list of dataset prediction
#     for i, row in enumerate(prediction[run]):
#         # convert conv to str
#         conv = row["history"][1:]
#         conv_str = "\n\n".join([{"user": "vsl", "assistant": "txt"}[e["role"]] + ": " + e["content"] for e in conv])
#         tf = row["answer"] == ground_truth[i]["ground_truth"]
#         dataset_preds.append([conv_str, tf])
#     prediction_dict[name] = dataset_preds

def majority_voting(prediction, ground_truth):
    majority = []
    count = 0
    for i, row in enumerate(ground_truth):
        preds = []
        for p in prediction:
            assert p[i]["history"][-1]["role"] == "assistant"
            preds.append(p[i]["history"][-1]["content"])
        # raw output
        answers = []
        for p in preds:
            answer = extract_after_answer(p).lstrip(":").rstrip(".").strip()
            answer = extract_mcq_option(answer)
            answers.append(answer)
        maj = majority_vote(answers)
        tf = maj == ground_truth[i]["ground_truth"]
        majority.append([maj, tf])
        if tf:
            count += 1
    print(count, "/", len(ground_truth), count / len(ground_truth))

if __name__ == "__main__":

    with open(predict_file, "r") as f:
        prediction = json.load(f)

    ground_truth_file = test_sets[test_set]["test_file"]
    with open(ground_truth_file, "r") as f:
        ground_truth = json.load(f)

    runs = len(prediction)
    assert runs == 11
    majority_voting(prediction, ground_truth)
    # visualize_predictions(prediction_dict, ground_truth, output_path, img_dir)