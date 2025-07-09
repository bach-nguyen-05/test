import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import copy
import multiprocessing as mp
import json
from multiprocessing import Queue, Process
from shared.conv_sampling import batch_process_predictions
from evaluation.vllm_wrapper import llama31_vllm_wrapper, llama32_vllm_wrapper, AsyncVLLMClient
from shared.utils import initialize_msg, initialize_e2e_msg
from shared.prompt import e2e_system_prompt
import vllm    

if __name__ == "__main__":
    # Load your evaluation samples
    os.environ["VLLM_USE_V1"] = "0"

    model_name = "llama3.2_e2e_original"
    vsl_model_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    test_set = "613"
    port=41651
    self_consistency = True

    setting = f"{model_name}_{test_set}_sc={self_consistency}"

    test_sets = {
        "724": {
            "test_file": "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/blind-vlm-project/datasets/eval/v5_val_724.json",
            "img_dir": "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/blind-vlm-project/datasets/eval/v5_eval"
        },
        "613": {
            "test_file": "/mnt/shared/shijie/blind-vlm-project/perturbed_613.json",
            "img_dir": "/mnt/shared/shijie/blind-vlm-project/"
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

    sample_rate = 11 if self_consistency else 1
    save_path = f"results/{setting}.json"
    timeout = 0.1

    # Load the samples
    with open(test_file, "r") as f:
        samples = json.load(f)

    for sample in samples:
        path = sample["path"]
        sample["path"] = os.path.join(img_dir, path)

    vsl_msgs = []
    
    for sample in samples:
        question = sample["question"]
        img_path = sample["path"]
        vsl_msg = initialize_e2e_msg(question, img_path, encode_images=True)
        vsl_msgs.append(vsl_msg)

    vsl_model = AsyncVLLMClient(
        model_path=vsl_model_path,
        port=port
    )

    import time
    start_time = time.time()

    all_results = []
    for trial in range(sample_rate):
        vsl_sampling_params = vllm.SamplingParams(
            temperature=1.0 if self_consistency else 0.0,
            seed=trial,
            max_tokens=128
        )
        results = batch_process_predictions(
            msgs=vsl_msgs,
            model=vsl_model,
            sampling_params=vsl_sampling_params,
            batch_size=64,
        )
        # all_results.append(results)

        sample = []
        assert len(vsl_msgs) == len(results)
        for idx, (res, msg) in enumerate(zip(results, vsl_msgs)):
            conv = [{"role": "assistant", "content": res}]
            sample.append({
                "cid": idx,
                "rounds": 1,
                "history": conv
            })
        all_results.append(sample)
        # Save to file
        with open(save_path, "w") as out:
            json.dump(all_results, out, indent=4)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("Time out", timeout)
    print("Done.")
