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
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

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


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
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
        max_num_batched_tokens = int(self.config.get('max_num_batched_tokens', 8192))

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
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
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
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
        if self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

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
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            response = output[0].to(idx.device)
            # log_probs = output[1].to(idx.device)

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                # log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

            # utilize current sampling params
            if self.sampling_params.n > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
                batch_size = batch_size * self.sampling_params.n
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

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
        if self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch)

# # Asynchronous VLLM API client
# from openai import OpenAI

# class AsyncVLLMClient:
#     def __init__(self, model_path: str, port: int = 8000, timeout: int = 30):
#         self.model_path = model_path
#         self.client = OpenAI(
#             base_url=f"http://0.0.0.0:{port}/v1",
#             api_key="BLIND_VQA",
#         )

#     async def predict(self, msg, **pred_args):
#         response = self.client.chat.completions.create(
#             model=self.model_path,
#             messages=msg,
#             **pred_args
#         )
#         return response.choices[0].message.content


# class vLLMConversationRollout(BaseRollout):

#     def __init__(self, 
#         actor_module: nn.Module, 
#         config: DictConfig, 
#         tokenizer, 
#         model_hf_config, 
#         external_model_client=None,
#         max_rounds=15,
#         **kwargs):
#         """A vLLM rollout. It requires the module is supported by the vllm.

#         Args:
#             module: module here follows huggingface APIs
#             config: DictConfig
#             tokenizer: the task/model tokenizer
#             model_hf_config: the huggingface config to initiallize the generating model in vllm
#             **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
#         """
#         super().__init__()
#         self.config = config
#         assert not (not config.enforce_eager and config.free_cache_engine), \
#             "disable CUDA graph (enforce_eager = False) if free cache engine"

#         tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
#         assert tensor_parallel_size <= torch.distributed.get_world_size(), \
#             "tensor parallel size should be less than or equal to the world size"
#         max_num_batched_tokens = int(self.config.get('max_num_batched_tokens', 8192))

#         if kwargs.get('train_tp', None) is not None:
#             # deployed with megatron
#             import os
#             os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
#             os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
#             train_tp = kwargs.get('train_tp', None)
#             num_tp_per_train_tp = train_tp // tensor_parallel_size
#             if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
#                 vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
#                                                   num_tp_per_train_tp=num_tp_per_train_tp)

#         assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
#             "model context length should be greater than total sequence length"

#         max_model_len = self.config.max_model_len if self.config.max_model_len \
#                         else config.prompt_length + config.response_length
#         max_model_len = int(max_model_len)

#         if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
#             raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
#                              please increase max_num_batched_tokens or disable chunked prefill')

#         self.inference_engine = LLM(
#             actor_module,
#             tokenizer=tokenizer,
#             model_hf_config=model_hf_config,
#             tensor_parallel_size=tensor_parallel_size,
#             dtype=config.dtype,
#             enforce_eager=config.enforce_eager,
#             gpu_memory_utilization=config.gpu_memory_utilization,
#             skip_tokenizer_init=False,
#             max_model_len=max_model_len,
#             load_format=config.load_format,
#             disable_log_stats=config.disable_log_stats,
#             max_num_batched_tokens=max_num_batched_tokens,
#             enable_chunked_prefill=config.enable_chunked_prefill,
#         )

#         # Offload vllm model to reduce peak memory usage
#         self.inference_engine.offload_model_weights()

#         kwargs = dict(
#             n=1,
#             logprobs=0,  # can be set to 0 and let actor to recompute
#             max_tokens=config.response_length,
#         )

#         # we may detokenize the result all together later
#         if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
#             kwargs['detokenize'] = False

#         # supporting adding any sampling params from the config file
#         for k in config.keys():
#             if hasattr(SamplingParams(), str(k)):
#                 kwargs[k] = config.get(k)

#         print(f"kwargs: {kwargs}")
#         self.sampling_params = SamplingParams(**kwargs)

#         self.pad_token_id = tokenizer.pad_token_id

#         if self.external_model_client is None:
#             self.external_model_client = AsyncVLLMClient(
#                 model_path="meta-llama/Llama-3.2-11B-Vision-Instruct",
#                 port=41651,
#             )
#         self.max_rounds = max_rounds

#     @contextmanager
#     def update_sampling_params(self, **kwargs):
#         # update sampling params
#         old_sampling_params_args = {}
#         if kwargs:
#             for key, value in kwargs.items():
#                 if hasattr(self.sampling_params, key):
#                     old_value = getattr(self.sampling_params, key)
#                     old_sampling_params_args[key] = old_value
#                     setattr(self.sampling_params, key, value)
#         yield
#         # roll back to previous sampling params
#         # if len(old_sampling_params_args):
#         for key, value in old_sampling_params_args.items():
#             setattr(self.sampling_params, key, value)

#     def _call_external_model(self, messages):
#         """Call the external model to generate a response."""
#         # Sampling parameters for the external API
#         sampling_dict = {
#             "max_tokens": 128,  # hard-coded for now
#             "temperature": 0.,  # temperature should be 0 for VLM
#             "top_p": getattr(self.sampling_params, "top_p", 1.0),
#             "top_k": getattr(self.sampling_params, "top_k", -1),
#         }
        
#         # Call the external model
#         response_text = self.external_model_client.predict(messages, **sampling_dict)
        
#         # Convert response to token IDs
#         response_ids = self.tokenizer.encode(response_text, return_tensors="pt")[0]
        
#         return response_text, response_ids

#     def _has_reached_answer(self, text):
#         """Check if the text contains an answer statement."""
#         return text.strip().startswith("The answer is")

#     def _tokenize_and_pad(self, text, max_length):
#         """Tokenize text and pad to max_length."""
#         tokens = self.tokenizer.encode(text, return_tensors="pt")[0]
#         if len(tokens) < max_length:
#             padding = torch.full((max_length - len(tokens),), self.pad_token_id, dtype=tokens.dtype)
#             tokens = torch.cat([tokens, padding], dim=0)
#         else:
#             tokens = tokens[:max_length]
#         return tokens

#     @torch.no_grad()
#     def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
#         # rebuild vllm cache engine
#         if self.config.free_cache_engine:
#             self.inference_engine.init_cache_engine()

#         idx = prompts.batch['input_ids']  # (bs, prompt_length)
#         # left-padded attention_mask
#         attention_mask = prompts.batch['attention_mask'] 
#         position_ids = prompts.batch['position_ids']
        
#         # used to construct attention_mask
#         eos_token_id = prompts.meta_info['eos_token_id']
        
#         batch_size = idx.size(0)
#         device = idx.device
        
#         # Set up sampling parameters
#         do_sample = prompts.meta_info.get('do_sample', True)
#         is_validate = prompts.meta_info.get('validate', False)
        
#         if not do_sample:
#             kwargs = {
#                 'best_of': 1,
#                 'top_p': 1.0,
#                 'top_k': -1,
#                 'min_p': 0.0,
#                 'temperature': 0,
#                 'n': 1  # if greedy, only 1 response
#             }
#         elif is_validate:
#             kwargs = {
#                 'top_k': self.config.val_kwargs.top_k,
#                 'top_p': self.config.val_kwargs.top_p,
#                 'temperature': self.config.val_kwargs.temperature,
#                 'n': 1,  # if validate, already repeat in ray_trainer
#             }

#         # Process each item in the batch
#         all_responses = []
#         all_complete_conversations = []
        
#         for b_idx in range(batch_size):
#             # Initialize conversation history for this batch item
#             conversation_history = []
            
#             # Get initial prompt text
#             prompt_tokens = _pre_process_inputs(self.pad_token_id, idx[b_idx])
#             prompt_text = self.tokenizer.decode(prompt_tokens)
            
#             # Add initial prompt to conversation history
#             conversation_history.append({"role": "system", "content": prompt_text})
            
#             # Start conversation loop
#             current_round = 0
#             reached_answer = False
            
#             while current_round < self.max_rounds and not reached_answer:
#                 # Format conversation for current model
#                 idx_list = [prompt_tokens]  # Convert to format expected by vLLM
                
#                 # Generate response from current model
#                 with self.update_sampling_params(**kwargs):
#                     output = self.inference_engine.generate(
#                         prompts=None,
#                         sampling_params=self.sampling_params,
#                         prompt_token_ids=idx_list,
#                         use_tqdm=False)
                    
#                     current_response = output[0].to(device)
#                     current_response_text = self.tokenizer.decode(current_response[0])
                
#                 # Check if current model has reached an answer
#                 if self._has_reached_answer(current_response_text):
#                     reached_answer = True
                
#                 # Add current model's response to conversation history
#                 conversation_history.append({"role": "assistant", "content": current_response_text})
                
#                 # If we haven't reached an answer and haven't hit max rounds, get external model response
#                 if not reached_answer and current_round < self.max_rounds - 1:
#                     # Get response from external model
#                     external_response_text, external_response_ids = self._call_external_model(conversation_history)
                    
#                     # Add external model's response to conversation history
#                     conversation_history.append({"role": "user", "content": external_response_text})
                    
#                     # Update prompt for next round
#                     prompt_text = self.tokenizer.decode(prompt_tokens) + "\n" + current_response_text + "\n" + external_response_text
#                     prompt_tokens = self.tokenizer.encode(prompt_text, return_tensors="pt")[0].tolist()
                
#                 current_round += 1
            
#             # Store final response and conversation history
#             all_responses.append(current_response)
#             all_complete_conversations.append(conversation_history)
        
#         # Process all responses into the expected format
#         responses = torch.stack(all_responses)
#         if responses.shape[1] < self.config.response_length:
#             responses = pad_sequence_to_length(responses, self.config.response_length, self.pad_token_id)
        
#         # Handle case of multiple samples if needed
#         if self.sampling_params.n > 1 and do_sample:
#             idx = idx.repeat_interleave(self.sampling_params.n, dim=0)
#             attention_mask = attention_mask.repeat_interleave(self.sampling_params.n, dim=0)
#             position_ids = position_ids.repeat_interleave(self.sampling_params.n, dim=0)
#             batch_size = batch_size * self.sampling_params.n
        
#         # Combine prompts and responses
#         seq = torch.cat([idx, responses], dim=-1)
        
#         # Update position IDs and attention mask
#         response_length = responses.size(1)
#         delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
#         delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
#         response_position_ids = position_ids[:, -1:] + delta_position_id
#         position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
#         response_attention_mask = get_eos_mask(response_id=responses, eos_token=eos_token_id, dtype=attention_mask.dtype)
#         attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
#         # Create batch with all data
#         batch = TensorDict(
#             {
#                 'prompts': idx,
#                 'responses': responses,
#                 'input_ids': seq,
#                 'attention_mask': attention_mask,
#                 'position_ids': position_ids,
#                 'conversations': all_complete_conversations  # Store the full conversation history
#             },
#             batch_size=batch_size)
        
#         # Free vllm cache engine if needed
#         if self.config.free_cache_engine:
#             self.inference_engine.free_cache_engine()
        
#         return DataProto(batch=batch)