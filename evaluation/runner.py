import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import multiprocessing as mp
import json
from multiprocessing import Queue, Process
from shared.conv_sampling import batch_process_conversations
from evaluation.vllm_wrapper import llama31_vllm_wrapper, llama32_vllm_wrapper, AsyncVLLMClient, human_reasoning_wrapper
from shared.utils import initialize_msg
import vllm
import requests

if __name__ == "__main__":
    # Set this flag to True to use human reasoning
    use_human_reasoning = True

    # mp.set_start_method("spawn", force=True)
    mp.set_start_method("fork", force=True)

    # Load your evaluation samples
    os.environ["VLLM_USE_V1"] = "0"

    txt_model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_name = "qwen2.5_7B_original"

    test_set = "spubench_500"
    self_consistency = False  # Set to FALSE for self-consistency evaluation

    setting = f"{model_name}_{test_set}_sc={self_consistency}"

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

    test_file = test_sets[test_set]["test_file"]
    img_dir = test_sets[test_set]["img_dir"]

    sample_rate = 11 if self_consistency else 1  # Sample rate is set to 1 because self-consistency=False
    vsl_model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    save_path = f"results/{setting}.json"
    timeout = 0.1

    # Load the samples
    with open(test_file, "r") as f:
        samples = json.load(f)

    for sample in samples:
        path = sample["path"]
        sample["path"] = os.path.join(img_dir, path)

    # Configure for human reasoning mode
    if use_human_reasoning:
        # LIMIT THE NUMBER OF SAMPLES FOR HUMAN REASONING
        txt_batch_size = 1     
        vis_batch_size = 1          
    else:
        txt_batch_size = 64
        vis_batch_size = 64

    txt_msgs = []
    vsl_msgs = []
    
    for sample in samples:
        question = sample["question"]
        img_path = sample["path"]
        txt_msg, vsl_msg = initialize_msg(question, img_path, encode_images=True)
        txt_msgs.append(txt_msg)
        vsl_msgs.append(vsl_msg)

    txt_kwargs = {
        "max_model_len": 8192,
        "max_num_seqs": 64,
        "gpu_memory_utilization": 0.4,
        "dtype": "bfloat16",
    }
    vsl_kwargs = {
        "max_model_len": 1024 + 256,
        "max_num_seqs": 64,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.45,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "disable_custom_all_reduce": True,
        "compilation_config": 0,
        "limit_mm_per_prompt": {"image": 1}
    }

    # Use human wrapper for text model if enabled
    if use_human_reasoning:
        txt_model = human_reasoning_wrapper()
    else:
        txt_model = llama31_vllm_wrapper(txt_model_path, **txt_kwargs)

    vsl_model = AsyncVLLMClient(
        model_path=vsl_model_path,
        port=41651
    )

    import time
    start_time = time.time()

    all_results = []
    for trial in range(sample_rate):
        # Run the multi-process pipeline
        txt_sampling_params = vllm.SamplingParams(
            temperature=1.0 if self_consistency else 0.0,
            max_tokens=512,
            seed=trial
        )
        vsl_sampling_params = vllm.SamplingParams(
            temperature=0.0,
            max_tokens=128
        )
        results = batch_process_conversations(
            txt_msgs,
            vsl_msgs,
            txt_model,
            vsl_model,
            txt_sampling_params,
            vsl_sampling_params,
            max_rounds=12,
            txt_batch_size=txt_batch_size, # 1 if use_human_reasoning else 64,
            vis_batch_size=vis_batch_size,
            timeout=timeout,
            conv_round_prompt=False,
        )
        all_results.append(results)

        # Save to file
        with open(save_path, "w") as out:
            json.dump(all_results, out, indent=4)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("Time out", timeout)
    print("Done.")