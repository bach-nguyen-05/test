import vllm
from PIL import Image
import transformers

DEFAULT_TXT_SAMPLE_PARAMS = vllm.SamplingParams(
    temperature=1.0,
    max_tokens=512
)

DEFAULT_VSL_SAMPLE_PARAMS = vllm.SamplingParams(
    temperature=0.0,
    max_tokens=128,
)

class llama31_vllm_wrapper:
    def __init__(self, model_path, **model_kwargs):
        self.model_path = model_path
        self.model = vllm.LLM(
            model=model_path,
            tokenizer=model_path,
            **model_kwargs
        )

    def predict(self, msg, sampling_params=DEFAULT_TXT_SAMPLE_PARAMS):
        prediction = self.model.chat(msg, sampling_params)
        ans = []
        for pred in prediction:
            ans.append(pred.outputs[0].text)
        return ans

class llama32_vllm_wrapper:
    def __init__(self, model_path, **model_kwargs):
        self.model_path = model_path
        self.model = vllm.LLM(
            model=model_path,
            tokenizer=model_path,
            **model_kwargs
        )
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
        prediction = self.model.generate(input_msg, sampling_params)
        for pred in prediction:
            ans.append(pred.outputs[0].text)
        return ans

import asyncio
from openai import OpenAI, AsyncOpenAI

class AsyncVLLMClient:
    def __init__(self, model_path: str, port: int = 8000, api_key: str = "BLIND_VQA"):
        self.model_path = model_path
        self.base_url = f"http://0.0.0.0:{port}/v1"
        self.api_key = api_key

    async def predict_msg(self, msg, sampling_params):
        temperature = getattr(sampling_params, "temperature", 0)
        max_tokens = getattr(sampling_params, "max_tokens", 128)
        async with AsyncOpenAI(base_url=self.base_url, api_key=self.api_key) as client:
            resp = await client.chat.completions.create(
                model=self.model_path,
                messages=msg,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content

    async def predict_batch(self, msgs, sampling_params):
        tasks = [self.predict_msg(m, sampling_params) for m in msgs]
        return await asyncio.gather(*tasks)

    # Original synchronous predict method
    def predict(self, msgs, sampling_params):
        return asyncio.run(self.predict_batch(msgs, sampling_params))
    
    # Minor change to support web UI
    async def predict_webui(self, msgs, sampling_params):
        return await self.predict_batch(msgs, sampling_params)

# class human_reasoning_wrapper:
#     def __init__(self, model_path="human", **model_kwargs):
#         self.model_path = model_path

#     def predict(self, msgs, sampling_params=None):
#         ans = []
#         for msg in msgs:
#             response = input("Your reasoning: ")
#             ans.append(response)
#         return ans

class human_reasoning_wrapper:
    def __init__(self, websocket=None, **kwargs):
        self.model_path = "human"

        ### Initialize for web API mode ###
        self.websocket = websocket
        if websocket:
            from queue import Queue
            self.req_q = Queue()
            self.res_q = Queue()

    def predict(self, msgs, sampling_params=None):
        ### Web API mode ###
        if self.websocket:
            # WebSocket mode with queue communication
            answers = []
            for msg in msgs:
                self.req_q.put({
                    "type": "reasoning_request",
                    "conversation": msg
                })
                response = self.res_q.get()
                answers.append(response)
            return answers


        ### Original synchronous mode ###
        else:
            ans = []
            for msg in msgs:
                response = input("Your reasoning: ")
                ans.append(response)
            return ans
