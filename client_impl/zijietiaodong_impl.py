

import os
import time

import llm_client_base

# pip install "volcengine-python-sdk[ark]"
from volcenginesdkarkruntime import AsyncArk

# config from .env
# VOLC_ACCESSKEY
# VOLC_SECRETKEY


class Zijietiaodong_Client(llm_client_base.LlmClientBase):
    def __init__(self):
        super().__init__()
        ak = os.getenv('VOLC_ACCESSKEY')
        sk = os.getenv('VOLC_SECRETKEY')
        assert ak is not None
        self.client = AsyncArk(ak=ak, sk=sk)

    async def chat_stream_async(self, model_name, history, temperature, force_calc_token_num):

        result_buffer = ''
        usage = None
        role = 'assistant'
        response_headers = None

        start_time = time.time()
        first_token_time = None

        async with self.client.chat.completions.with_streaming_response.create(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
        ) as resp:

            async for chunk_resp in resp.iter_text():
                print(chunk_resp)
                chunk = chunk_resp.body
                usage = chunk['usage']
                response_headers = chunk_resp.headers
                result_buffer += chunk['result']
                if chunk['result'] and first_token_time is None:
                    first_token_time = time.time()

                yield {
                    'role': role,
                    'delta_content': chunk['result'],
                    'accumulated_content': result_buffer,
                }

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': 0,
            'usage': usage,
            'first_token_time': first_token_time - start_time,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Zijietiaodong_Client()
    model_name = "Doubao-lite-128k"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
