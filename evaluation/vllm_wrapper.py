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

    async def predict(self, msgs, sampling_params):
        return await self.predict_batch(msgs, sampling_params)

class human_reasoning_wrapper:
    def __init__(self, **kwargs):
        """Enhanced human reasoning wrapper with better interaction"""
        self.model_path = "human"
        self.conversation_count = 0
        self.current_challenge = 0

    def predict(self, msgs, sampling_params=None):
        ans = []
        for i, msg in enumerate(msgs):
            # Detect if this is a new challenge (contains original question)
            is_new_challenge = self._is_new_challenge(msg)
            if is_new_challenge:
                self.current_challenge += 1
                self.conversation_count = 1  # Reset round for new challenge
                print(f"STARTING CHALLENGE {self.current_challenge}/{len(msgs)}")
            else:
                self.conversation_count += 1
            print(f"Round: {self.conversation_count}")

            # Show conversation history with better formatting
            self._display_conversation_history(msg, is_new_challenge)

            # Get user input
            user_response = self._get_user_input()
            results.append(user_response)
            print(f"Response Received\n")
        return ans

    def _is_new_challenge(self, msg):
        """Check if this is a new challenge (contains original question)"""
        for turn in msg:
            if turn['role'] == 'user' and 'Select from the following choices:' in turn['content']:
                return True
        return False


    def _display_conversation_history(self, msg, show_original_question):
        """Display conversation history, showing the original question only if show_original_question is True."""
        shown_original_question = False
        for turn in msg:
            role = turn['role'].upper()
            content = turn['content']
            if role == 'USER' and 'Select from the following choices:' in content:
                if show_original_question and not shown_original_question:
                    if len(content) > 300:
                        content = content[:300] + "\n... [truncated] ..."
                    print(f"\n{role}: {content}")
                    shown_original_question = True
                continue
            elif role == 'USER' and 'Select from the following choices:' not in content:
                print(f"\nVISUAL INTERPRETER: {content[:200]}..." if len(content) > 200 else f"\nVISUAL INTERPRETER: {content}")
            elif role == 'ASSISTANT':
                print(f"\nYOUR PREVIOUS RESPONSE: {content[:200]}..." if len(content) > 200 else f"\nYOUR PREVIOUS RESPONSE: {content}")
            elif role == 'SYSTEM':
                continue
            else:
                if len(content) > 300:
                    content = content[:300] + "\n... [truncated] ..."
                print(f"\n{role}: {content}")
                
    def _get_user_input(self):
        """Get validated multi-line input"""
        lines = []
        empty_count = 0
        
        while True:
            try:
                line = input()
                if line == "":
                    empty_count += 1
                    if empty_count >= 2 and lines:
                        break
                    elif empty_count == 1 and not lines:
                        print("Please enter some content first.")
                        continue
                else:
                    empty_count = 0
                    lines.append(line)
            except KeyboardInterrupt:
                print("\nExiting...")
                return "QUIT"
        
        response = "\n".join(lines).strip()
        return response if response else "I need more time to analyze this."