

import os
import time

import llm_client_base

import dashscope

# config from .env
# DASHSCOPE_API_KEY


class Alibaba_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self):
        super().__init__()

        api_key = os.getenv('DASHSCOPE_API_KEY')
        assert api_key is not None

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        temperature = model_param['temperature']

        start_time = time.time()
        stream_response = await dashscope.AioGeneration.call(
            model=model_name,
            messages=history,
            temperature=temperature,
            result_format='message',
            stream=True, incremental_output=True,
        )

        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async for chunk_resp in stream_response:
            chunk = chunk_resp['output']

            usage = chunk_resp['usage']
            if usage:
                usage = {
                    'prompt_tokens': usage['input_tokens'],
                    'completion_tokens': usage['output_tokens'],
                }

            choice0 = chunk['choices'][0]
            delta_data = choice0['message']['content']
            role = choice0['message']['role']
            finish_reason = choice0['finish_reason']

            if delta_data and first_token_time is None:
                first_token_time = time.time()

            result_buffer += delta_data
            yield {
                'role': role,
                'delta_content': delta_data,
                'accumulated_content': result_buffer,
                'usage': usage,
            }

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'usage': usage,
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Alibaba_Client()
    model_name = "qwen-plus"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
