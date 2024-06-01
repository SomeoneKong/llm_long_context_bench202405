

import os
import time

import llm_client_base

# pip install "volcengine-python-sdk[ark]"
from volcenginesdkarkruntime import AsyncArk

# config from .env
# VOLC_ACCESSKEY
# VOLC_SECRETKEY


class ByteDance_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self):
        super().__init__()
        ak = os.getenv('VOLC_ACCESSKEY')
        sk = os.getenv('VOLC_SECRETKEY')
        assert ak is not None
        self.client = AsyncArk(ak=ak, sk=sk)

    async def close(self):
        await self.client._client.aclose()

    async def chat_stream_async(self, model_name, history, temperature, force_calc_token_num):
        start_time = time.time()

        resp = await self.client.chat.completions.create(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
            stream_options={'include_usage': True},
        )

        result_buffer = ''
        usage = None
        role = None
        real_model_name = None
        finish_reason = None
        first_token_time = None

        async for chunk in resp:
            if chunk.usage:
                usage = {
                    'prompt_tokens': chunk.usage.prompt_tokens,
                    'completion_tokens': chunk.usage.completion_tokens,
                }

            if chunk.choices:
                choice0 = chunk.choices[0]
                result_buffer += choice0.delta.content
                role = choice0.delta.role
                if choice0.finish_reason:
                    finish_reason = choice0.finish_reason
                if chunk.model:
                    real_model_name = chunk.model

                if choice0.delta.content and first_token_time is None:
                    first_token_time = time.time()

                yield {
                    'role': role,
                    'delta_content': choice0.delta.content,
                    'accumulated_content': result_buffer,
                }

        await resp.close()

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'usage': usage,
            'first_token_time': first_token_time - start_time,
            'completion_time': completion_time - start_time,
            'model_name': real_model_name,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = ByteDance_Client()
    model_name = "ep-xxxxxxxxx"  # "Doubao-lite-128k"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
