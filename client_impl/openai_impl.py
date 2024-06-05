

import os
import time

import llm_client_base

from openai import AsyncOpenAI

# config from .env
# OPENAI_API_KEY
# HTTP_PROXY
# HTTPS_PROXY

# os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
# os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"


class OpenAI_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self,
                 api_base_url=None,
                 api_key=None,
                 ):
        super().__init__()
        self.client = AsyncOpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, temperature, force_calc_token_num):
        start_time = time.time()

        system_fingerprint = None
        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async with await self.client.chat.completions.create(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
            stream_options={'include_usage': True},
        ) as response:

            async for chunk in response:
                system_fingerprint = chunk.system_fingerprint
                if chunk.choices:
                    finish_reason = chunk.choices[0].finish_reason
                    delta_info = chunk.choices[0].delta
                    if delta_info.role:
                        role = delta_info.role
                    if delta_info.content:
                        result_buffer += delta_info.content

                        if first_token_time is None:
                            first_token_time = time.time()

                        yield {
                            'role': role,
                            'delta_content': delta_info.content,
                            'accumulated_content': result_buffer,
                        }
                if chunk.usage:
                    usage = chunk.usage.dict()


        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'system_fingerprint': system_fingerprint,
            'usage': usage or {},
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }

    async def close(self):
        await self.client.close()


if __name__ == '__main__':
    import asyncio
    import os

    client = OpenAI_Client()
    model_name = "gpt-3.5-turbo"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
