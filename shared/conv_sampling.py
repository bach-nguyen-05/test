import multiprocessing as mp
from multiprocessing import Process, Queue
import threading

from dataclasses import dataclass, field
from typing import Any, Dict, List
import time
import copy

# Utility functions
from shared.utils import *

# Sentinel to signal shutdown
SENTINEL = None

txt_model_global = None
vis_model_global = None

# For thread safety
model_lock = threading.Lock()

@dataclass
class Conversation:
    cid: int
    txt_history: List[Any] = field(default_factory=list)
    vis_history: List[Any] = field(default_factory=list)
    rounds: int = 0
    answer: str = ""

def check_msg_format(msg: List[Dict[str, Any]]) -> bool:
    """
    Check if the message format is correct.
    """
    if not isinstance(msg, list):
        print("Messages is not a list.")
        return False
    for m in msg:
        if not isinstance(m, list):
            print("Message is not a list.")
            return False
        for i in range(len(m)):
            if not isinstance(m[i], dict):
                print("Message is not a dict.")
                return False
            if "role" not in m[i] or "content" not in m[i]:
                print("Message does not contain role or content.")
                return False
            if m[i]["role"] not in ["user", "assistant", "system"]:
                print("Role is not user, assistant or system.")
                return False

# Worker functions must be at module level for 'spawn' compatibility
def text_worker(txt_q: Queue, mid_q: Queue, batch_size=64, timeout=5, sampling_params: Dict = None):
    """
    Text worker process: loads model inside process and processes prompts.
    """
    global txt_model_global
    model = txt_model_global

    buffer = []
    last_process_time = time.time()

    def process_buffer():
        nonlocal buffer, last_process_time
        if not buffer:
            return
        
        cids, prompts = zip(*buffer)
        prompts = list(prompts)
        print("text model", len(prompts))
        check_msg_format(prompts)
        # with model_lock:
        resp = model.predict(prompts, sampling_params=sampling_params)
        for cid, r in zip(cids, resp):
            mid_q.put((cid, r, 'text'))
        buffer.clear()
        last_process_time = time.time()

    while True:
        try:
            item = txt_q.get(timeout=timeout)
            if item is SENTINEL:
                process_buffer()
                break
            buffer.append(item)

            if len(buffer) >= batch_size:
                process_buffer()
            elif time.time() - last_process_time > timeout and buffer:
                process_buffer()

        except Exception:
            if buffer:
                # Process remaining items
                process_buffer()
                
    if buffer:
        process_buffer()
    

def vision_worker(vis_q: Queue, mid_q: Queue, batch_size=64, timeout=5, sampling_params: Dict = None):
    """
    Vision worker process: loads model inside process and processes prompts.
    """
    global vis_model_global
    model = vis_model_global

    buffer = []
    last_process_time = time.time()

    def process_buffer():
        nonlocal buffer, last_process_time
        if not buffer:
            return
        
        cids, prompts = zip(*buffer)
        prompts = list(prompts)
        print("vision model", len(prompts))
        check_msg_format(prompts)
        # with model_lock:
        resp = model.predict(prompts, sampling_params=sampling_params)
        for cid, r in zip(cids, resp):
            mid_q.put((cid, r, 'vision'))
        buffer.clear()
        last_process_time = time.time()

    while True:
        try:
            item = vis_q.get(timeout=timeout)
            if item is SENTINEL:
                process_buffer()
                break
            buffer.append(item)

            if len(buffer) >= batch_size:
                process_buffer()
            elif time.time() - last_process_time > timeout and buffer:
                process_buffer()
        except Exception:
            if buffer:
                # Process remaining items
                process_buffer()
    if buffer:
        process_buffer()

def batch_process_conversations(
    txt_msgs: List[Any],
    vis_msgs: List[Any],
    txt_model,
    vis_model,
    txt_sampling_params: Dict,
    vis_sampling_params: Dict,
    max_rounds: int = 10,
    timeout: int = 0.1,
    txt_batch_size: int = 64,
    vis_batch_size: int = 64,
    conv_round_prompt: bool = False,
) -> List[Dict[str, Any]]:
    """
    Multiprocess pipeline:
      1) Spawn spawn-based processes for text and vision workers.
      2) First round: feed only to text.
      3) Alternate: text -> vision -> text until answer or max_rounds.
    """
    def prompt_template(round_num: int) -> str:
        reminder = "Every assistant turn must contain the four lines exactly in this order — Thought, Weights, Factors, Action."
        if round_num < max_rounds:
            return f"\nReminder: This is round {round_num}/{max_rounds}. {reminder}"
        else:
            return f"\nReminder: This is round {round_num}/{max_rounds}. {reminder} You must include your final answer in your response."

    from queue import Queue


    global txt_model_global, vis_model_global
    txt_model_global = txt_model
    vis_model_global = vis_model

    # Create queues
    txt_q = Queue()
    vis_q = Queue()
    mid_q = Queue()

    # Start worker processes
    p_txt = threading.Thread(target=text_worker, args=(txt_q, mid_q, txt_batch_size, timeout, txt_sampling_params))
    p_vis = threading.Thread(target=vision_worker, args=(vis_q, mid_q, vis_batch_size, timeout, vis_sampling_params))
    p_txt.start()
    p_vis.start()

    # Initialize conversations
    convs = {
        i: Conversation(i, txt_history=copy.deepcopy(txt_msgs[i]), vis_history=copy.deepcopy(vis_msgs[i]))
        for i in range(len(txt_msgs))
    }
    
    if conv_round_prompt:
        for conv in convs.values():
            assert len(conv.txt_history) == 2
            assert conv.txt_history[-1]['role'] == 'user'
            conv.txt_history[-1]['content'] = conv.txt_history[-1]['content'] + prompt_template(1)

    # First round: send only to text
    for conv in convs.values():
        txt_q.put((conv.cid, conv.txt_history))

    active = set(convs.keys())
    while active:
        cid, output, who = mid_q.get()
        conv = convs[cid]

        if who == 'text':
            conv.rounds += 1
            conv.txt_history.append({"role":"assistant","content": output})
            # Check end condition
            if is_answer_found(remove_possible_cot(output)):
                answer = extract_after_answer(output).lstrip(":").rstrip(".").strip()
                conv.answer = extract_mcq_option(answer)
                active.remove(cid)
            elif conv.rounds >= max_rounds:
                conv.answer = None
                active.remove(cid)
            else:
                # feed to vision next
                conv.vis_history.append({"role":"user","content": remove_possible_cot(output)})  
                vis_q.put((cid, copy.deepcopy(conv.vis_history)))
        else:
            if conv_round_prompt:
                output += prompt_template(conv.rounds + 1)
            conv.txt_history.append({"role":"user","content": output})
            conv.vis_history.pop()
            # feed back to text
            txt_q.put((cid, copy.deepcopy(conv.txt_history)))

    # Shutdown workers
    txt_q.put(SENTINEL)
    vis_q.put(SENTINEL)
    p_txt.join()
    p_vis.join()

    results = []
    # Sort results by cid
    for i in range(len(txt_msgs)):
        c = convs[i]
        results.append({
            'cid':    c.cid,
            'rounds': c.rounds,
            'history': c.txt_history,
            'answer':  c.answer,
            'status':  'answered' if c.answer else 'timeout'
        })
    cids = [r['cid'] for r in results]
    assert cids == list(range(len(results))), f"Out-of-order! got {cids}"

    # Gather results
    return results


def batch_process_predictions(
    msgs: List[Any],
    model,
    sampling_params: Dict,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    batch = []
    results = []
    for msg in msgs:
        batch.append(msg)
        if len(batch) == batch_size:
            outputs = model.predict(batch, sampling_params=sampling_params)
            for out in outputs:
                results.append(out)
            batch = []
    if batch:
        outputs = model.predict(batch, sampling_params=sampling_params)
        for out in outputs:
            results.append(out)
    return results