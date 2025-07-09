# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
import re
import copy

from shared.utils import initialize_msg, is_answer_found, remove_possible_cot
from shared.conv_sampling import batch_process_conversations
from evaluation.vllm_wrapper import DEFAULT_TXT_SAMPLE_PARAMS, DEFAULT_VSL_SAMPLE_PARAMS, AsyncVLLMClient
import transformers
from PIL import Image
import threading

# one lock per engine:
_current_model_lock  = threading.Lock()
_external_model_lock = threading.Lock()

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)

            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

def _to_object_array(list_of_lists):
    arr = np.empty(len(list_of_lists), dtype=object)
    for i, x in enumerate(list_of_lists):
        arr[i] = x
    return arr

def get_last_eos_mask(response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    '''
    Returns a mask that is 1 up to and including the last occurrence of any eos_token, and 0 after.
    
    end of sentence token can be int or list: 1 or [1, 2]
    e.g. eos_token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    last_eos_mask: [1, 1, 1, 1, 1, 1, 1, 0, 0]
    
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 1]
    last_eos_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1]
    '''
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    # Create a boolean mask where True indicates positions of any eos_token
    eos_positions = torch.zeros_like(response_id, dtype=torch.bool)
    for token in eos_token:
        eos_positions |= response_id.eq(token)
    
    # If no EOS tokens found, return all zeros
    if not eos_positions.any():
        return torch.zeros_like(response_id, dtype=dtype)
    
    # Find the last position where any eos_token appears
    # Need to handle batch dimension if present
    if response_id.dim() > 1:
        # For batched inputs
        batch_size = response_id.size(0)
        seq_len = response_id.size(1)
        last_eos_mask = torch.ones_like(response_id, dtype=dtype)
        
        for i in range(batch_size):
            if eos_positions[i].any():
                indices = eos_positions[i].nonzero().view(-1)
                last_eos_idx = indices[-1].item()  # Get the last occurrence
                if last_eos_idx + 1 < seq_len:
                    last_eos_mask[i, last_eos_idx + 1:] = 0
            else:
                # No EOS token found in this sequence
                last_eos_mask[i, :] = 0
    else:
        # For single sequence
        if eos_positions.any():
            indices = eos_positions.nonzero().view(-1)
            last_eos_idx = indices[-1].item()  # Get the last occurrence
            last_eos_mask = torch.ones_like(response_id, dtype=dtype)
            if last_eos_idx + 1 < response_id.size(0):
                last_eos_mask[last_eos_idx + 1:] = 0
        else:
            # No EOS token found
            last_eos_mask = torch.zeros_like(response_id, dtype=dtype)
    
    return last_eos_mask

class current_model_wrapper:
    def __init__(self, model, 
        format_conversation_for_model,
        ids_to_text):
        self.model = model
        self.format_conversation_for_model = format_conversation_for_model
        self.ids_to_text = ids_to_text

    def predict(self, msg, sampling_params=DEFAULT_TXT_SAMPLE_PARAMS):
        # print("before seed = ", sampling_params.seed)
        # possibly we can randomly pick another seed
        sampling_params = copy.deepcopy(sampling_params)
        sampling_params.seed = np.random.randint(1000000)
        if getattr(sampling_params, 'max_tokens', None) is None:
            sampling_params.max_tokens = 512
            print("max_tokens is None, set to 512")
        
        # print("after seed = ", sampling_params.seed)
        current_model_tokens_list = [self.format_conversation_for_model(conv) for conv in msg]
        vllm_input = [
            {
                'prompt_token_ids': current_model_tokens.tolist()
            } for current_model_tokens in current_model_tokens_list]
        with _current_model_lock:
            current_model_output = self.model.generate(
                prompts=vllm_input,
                sampling_params=sampling_params,
                use_tqdm=False
            )
        assert len(current_model_output) == len(msg)
        n = len(current_model_output)
        current_response_text_list = []
        for i in range(n):
            current_response_ids = current_model_output[i].outputs[0].token_ids
            current_response_text_list.append(self.ids_to_text(current_response_ids))
        return current_response_text_list

class external_model_wrapper:
    def __init__(self, model, model_path="meta-llama/Llama-3.2-11B-Vision-Instruct"):
        self.model = model
        self.processor = transformers.AutoProcessor.from_pretrained(model_path)

    def predict(self, msg, sampling_params=DEFAULT_VSL_SAMPLE_PARAMS):
        input_msg = []
        for m in msg:
            assert type(m[1]["content"]) == list
            assert "image_url" in m[1]["content"][0]
            assert "url" in m[1]["content"][0]["image_url"]
            url = m[1]["content"][0]["image_url"]["url"]
            m[1]["content"] = "<|image|>"

            input_text = self.processor.apply_chat_template(m, add_generation_prompt=True)
            try:
                # Open image and ensure it's in the right format (RGB)
                image = Image.open(url).convert("RGB")
                
                # Add to input message list
                input_msg.append({
                    "prompt": input_text,
                    "multi_modal_data": {"image": image}
                })
            except Exception as e:
                print(f"Error processing image {url}: {e}")
                return []

        for m in input_msg:
            assert "prompt" in m
            assert "multi_modal_data" in m
            assert "image" in m["multi_modal_data"]
            
        ans = []
        with _external_model_lock:
            prediction = self.model.generate(input_msg, sampling_params, use_tqdm=False)
        for pred in prediction:
            ans.append(pred.outputs[0].text)
        return ans

class vLLMConversationRollout(BaseRollout):
    def __init__(self, 
                 model_path: str, 
                 config: DictConfig, 
                 tokenizer, 
                 model_hf_config, 
                 external_model_path: str = None,
                 external_model_port: int = 8000,
                 max_rounds: int = 15,
                 **kwargs):
        """A vLLM rollout that supports conversations between the current model and an external model.

        Args:
            model_path: Path to the current model
            config: DictConfig
            tokenizer: The tokenizer for the model
            model_hf_config: The HuggingFace config for the model
            external_model_path: Path to the external model
            external_model_port: Port for the external model API
            max_rounds: Maximum number of conversation rounds
            **kwargs: Additional arguments
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_rounds = max_rounds
        
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        max_model_len = self.config.max_model_len if self.config.max_model_len \
                        else config.prompt_length + config.response_length
        max_model_len = int(max_model_len)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        print("max model len", max_model_len)

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            # tokenizer=model_path, # add this tokenizer param just in case
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        # probably no need to change any of these kwargs
        external_model_kwargs = {
            "max_model_len": 1024 + 256,
            "max_num_seqs": 16,
            "enforce_eager": True,
            "gpu_memory_utilization": 0.3,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "disable_custom_all_reduce": True,
            "compilation_config": 0,
            "limit_mm_per_prompt": {"image": 1},
            "tensor_parallel_size": tensor_parallel_size,
            "distributed_executor_backend": "external_launcher",
            # "enable_chunked_prefill": False,
            # "max_num_batched_tokens": 1024 + 256,
            # "enable_sleep_mode": True
        }
        print("loading external model")
        self.external_model = AsyncVLLMClient(
            model_path="meta-llama/Llama-3.2-11B-Vision-Instruct",
            port=41651
        )
        # self.external_model = LLM(
        #     "meta-llama/Llama-3.2-11B-Vision-Instruct",
        #     **external_model_kwargs
        # )
        print("external model loaded")

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def ids_to_text(self, token_ids):
        """Convert token IDs to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Filter out padding tokens
        if self.pad_token_id in token_ids:
            token_ids = token_ids[:token_ids.index(self.pad_token_id)]
            
        return self.tokenizer.decode(token_ids, skip_special_tokens=True) # don't need EOS token
    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        image_paths = prompts.non_tensor_batch["image_path"]
        
        # Make sure we have an image path for each sample
        assert len(image_paths) == batch_size, f"Number of image paths doesn't match batch size: {len(image_paths)} and {batch_size}"
        
        # After padding responses to the same length
        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            REPEAT_FACTOR = self.sampling_params.n

            # Initialize separate conversation histories for each model and each sample
            current_model_conversations = [[] for _ in range(batch_size)]
            external_model_conversations = [[] for _ in range(batch_size)]
            conversation_logs = [[] for _ in range(batch_size)]
            
            # Process initial prompts and initialize conversation histories
            for i, raw_prompt_ids in enumerate(non_tensor_batch['raw_prompt_ids']):
                # Try to extract conversation structure from initial prompt tokens
                initial_conversation = self.extract_conversation_from_tokens(
                    raw_prompt_ids,
                    start_marker=self.config.start_marker,
                    end_marker=self.config.end_marker,
                    role_end_marker=self.config.end_role_marker,
                )

                assert len(initial_conversation) == 2
                assert initial_conversation[1]["role"] == "user"

                question = initial_conversation[1]["content"]
                img_path = image_paths[i]

                cot = self.config.get('cot', False)
                current_model_msg, external_model_msg = initialize_msg(question, img_path, encode_images=True, cot=cot)
                current_model_conversations[i] = current_model_msg
                external_model_conversations[i] = external_model_msg
                
                # Add initial user query to conversation log
                for msg in initial_conversation:
                    if msg["role"] != "system":
                        # For non-system messages, add to the conversation log
                        role_mapping = {"user": "Human", "assistant": "Current Model"}
                        conversation_logs[i].append({
                            "speaker": role_mapping.get(msg["role"], msg["role"]),
                            "content": msg["content"]
                        })

            def _convert_external_conv(conv):
                conv = copy.deepcopy(conv)
                for m in conv:
                    if m["role"] == "assistant":
                        m["content"] = remove_possible_cot(m["content"])
                return conv

            final_responses_runs = [[] for _ in range(REPEAT_FACTOR)]
            current_model_conversations_history_runs = [[] for _ in range(REPEAT_FACTOR)]
            external_model_conversations_history_runs = [[] for _ in range(REPEAT_FACTOR)]

            for repeat_idx in range(REPEAT_FACTOR):
                txt_sampling_params = copy.deepcopy(self.sampling_params)
                txt_sampling_params.n = 1
                txt_sampling_params.seed = repeat_idx
                txt_sampling_params.max_tokens = 512

                vsl_sampling_params = SamplingParams(
                    temperature=0,
                    max_tokens=128
                )
                
                results = batch_process_conversations(
                    copy.deepcopy(current_model_conversations),
                    copy.deepcopy(external_model_conversations),
                    current_model_wrapper(self.inference_engine, self.format_conversation_for_model, self.ids_to_text),
                    # external_model_wrapper(self.external_model),
                    self.external_model,
                    txt_sampling_params,
                    vsl_sampling_params,
                    max_rounds=self.max_rounds,
                    txt_batch_size=64,
                    vis_batch_size=64,
                    timeout=0.1,
                )

                current_conv = [copy.deepcopy(r["history"]) for r in results]
                external_conv = [_convert_external_conv(r["history"]) for r in results]
                final_token_ids = [self.format_conversation_for_model(c).tolist() for c in current_conv]
                final_responses_runs[repeat_idx] = final_token_ids
                current_model_conversations_history_runs[repeat_idx] = current_conv
                external_model_conversations_history_runs[repeat_idx] = external_conv

            # interleave runs results
            final_responses = [x for pair in zip(*final_responses_runs) for x in pair]
            current_model_conversations_history = [x for pair in zip(*current_model_conversations_history_runs) for x in pair]
            external_model_conversations_history = [x for pair in zip(*external_model_conversations_history_runs) for x in pair]

        # Pad responses to the same length
        response = pad_2d_list_to_length(final_responses, self.pad_token_id,
                                         max_length=self.config.response_length).to(idx.device)

        # if self.sampling_params.n > 1 and not is_validate:
        if self.sampling_params.n > 1 and do_sample:
            repeat_factor = self.sampling_params.n

            non_tensor_batch['current_model_conversations'] = _to_object_array(current_model_conversations_history)
            non_tensor_batch['external_model_conversations'] = _to_object_array(external_model_conversations_history)
            # non_tensor_batch['conversation_logs'] = _to_object_array(conversation_logs_history)
            non_tensor_batch['raw_prompt_ids'] = _repeat_interleave(non_tensor_batch['raw_prompt_ids'], repeat_factor)
            
            idx = _repeat_interleave(idx, repeat_factor)
            attention_mask = _repeat_interleave(attention_mask, repeat_factor)
            position_ids = _repeat_interleave(position_ids, repeat_factor)

            # 3. Update the batch size
            new_batch_size = batch_size * repeat_factor
            
            # non_tensor_batch = new_non_tensor_batch
            batch_size = new_batch_size
        else:
            # For validation, just add the conversation histories without repetition
            non_tensor_batch['current_model_conversations'] = _to_object_array(current_model_conversations_history)
            non_tensor_batch['external_model_conversations'] = _to_object_array(external_model_conversations_history)
            # non_tensor_batch['conversation_logs'] = _to_object_array(conversation_logs_history)

        # Same post-processing as original vLLMRollout
        seq = torch.cat([idx, response], dim=-1)
        
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        # response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        
        # use self.config.end_marker as the eos_token
        eos_token = self.tokenizer.convert_tokens_to_ids(self.config.end_marker)
        response_attention_mask = get_last_eos_mask(response_id=response, eos_token=eos_token, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        # force every entry to a flat 1‑D object array
        for key, val in non_tensor_batch.items():
            non_tensor_batch[key] = _to_object_array(val)

        if 'image_path' in non_tensor_batch:
            del non_tensor_batch['image_path']

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _get_role_markers(self):
        test_string = "TEST_STRING"
        test_messages = [
            [{"role": "system", "content": test_string}],
            [{"role": "user", "content": test_string}],
            [{"role": "assistant", "content": test_string}]
        ]
        role_markers = {}
        for role, messages in zip(["system", "user", "assistant"], test_messages):
            # Use apply_chat_template to see how each role is formatted
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False)
            # Encode to get the token IDs for the formatted text
            formatted_ids = self.tokenizer.encode(formatted, add_special_tokens=True)
            # Get test content IDs to exclude from markers
            content_ids = self.tokenizer.encode(test_string, add_special_tokens=False)
            # Identify marker IDs by removing content IDs
            marker_ids = [id for id in formatted_ids if id not in content_ids]
            print("role marker before decode")
            print(formatted)
            print("role marker after decode")
            print(self.tokenizer.decode(marker_ids))
            print("-" * 20)
            role_markers[role] = marker_ids
        self.role_markers = role_markers
        self.test_string = test_string

    def extract_conversation_from_tokens(self, token_ids, start_marker="<|im_start|>", end_marker="<|im_end|>", role_end_marker=""):
        # self._get_role_markers()
        # Convert token IDs to tokens (this avoids issues if special tokens are split in the decoded string)
        # text = self.tokenizer.convert_ids_to_tokens(token_ids)
        text = self.tokenizer.decode(token_ids)
    
        # Pattern to find all segments between the markers.
        pattern = re.escape(start_marker) + r'(.*?)' + re.escape(end_marker)
        segments = re.findall(pattern, text, flags=re.DOTALL)
        
        conversation = []
        
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

    def format_conversation_for_model(self, conversation):
        """
        Format a conversation for the current model.
        Ensures the generation marker is added if needed.
        
        Args:
            conversation: List of message dicts with 'role' and 'content' fields
            
        Returns:
            token_ids: Tensor of token IDs formatted for the model
        """
        # First check if the last message is from the assistant role
        has_assistant_last = False
        if conversation and len(conversation) > 0:
            has_assistant_last = conversation[-1]["role"] == "assistant"
        
        # Clone the conversation to avoid modifying the original
        conversation_copy = conversation.copy()
        
        # Use the model's built-in chat template if available
        assert hasattr(self.tokenizer, "apply_chat_template")
        formatted_input = self.tokenizer.apply_chat_template(
            conversation_copy, 
            tokenize=False,
            add_generation_prompt=not has_assistant_last
        )
        token_ids = self.tokenizer.encode(formatted_input, return_tensors="pt")[0]
        return token_ids