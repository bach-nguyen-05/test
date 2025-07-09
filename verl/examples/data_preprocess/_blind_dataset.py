import os
import datasets
from datasets import Dataset
import json

from verl.utils.hdfs_io import copy, makedirs
import argparse

TXT_SYSTEM_PROMPT = '''You are a helpful assistant.'''.replace("\n", " ")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--debug', default=False)

    args = parser.parse_args()
    data_path = args.data_path
    debug_mode = args.debug

    dataset = {
        "conversations": [],
        "question": [],
        "answer": [],
        "image": []
    }
    with open(data_path) as f:
        data = json.load(f)
        for i, d in enumerate(data):
            if debug_mode and i >= 200:
                break

            dataset["answer"].append(d["answer"])
            dataset["image"].append(d["image"])
            dataset["question"].append(d["question"])
            dataset["conversations"].append([
                {
                    "role": "system",
                    "content": TXT_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": d["question"]
                }
            ])

    split_dataset = Dataset.from_dict(dataset).train_test_split(test_size=0.02, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    data_source = "blind500_new_reasoning_scheme"

    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')
            image = example.pop('image')
            solution = example.pop('answer')
            conversation = example.pop('conversations')
            data = {
                "data_source": data_source,
                "prompt": conversation,
                "image_path": image,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution,
                    "question": question_raw,
                    "image": image,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    if debug_mode:
        train_dataset.to_parquet(os.path.join(local_dir, 'train_debug.parquet'))
        test_dataset.to_parquet(os.path.join(local_dir, 'test_debug.parquet'))
    else:
        train_dataset.to_parquet(os.path.join(local_dir, 'train_mcq.parquet'))
        test_dataset.to_parquet(os.path.join(local_dir, 'test_mcq.parquet'))
