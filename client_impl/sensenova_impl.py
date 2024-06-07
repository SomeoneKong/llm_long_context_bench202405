

import os
import time

import llm_client_base

# pip install sensenova
import sensenova

# config from .env
# SENSENOVA_KEY_ID
# SENSENOVA_SECRET_KEY


class SenseNova_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self):
        super().__init__()

        key_id = os.getenv('SENSENOVA_KEY_ID')
        secret_access_key = os.getenv('SENSENOVA_SECRET_KEY')
        assert key_id is not None

        sensenova.access_key_id = key_id
        sensenova.secret_access_key = secret_access_key

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        temperature = model_param['temperature']

        start_time = time.time()
        response = await sensenova.ChatCompletion.acreate(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
            max_new_tokens=2047,
        )

        result_buffer = ''
        usage = None
        role = None
        finish_reason = None
        first_token_time = None

        async for chunk_resp in response:
            if chunk_resp.status.code != 0 and chunk_resp.data.choices[0].finish_reason == 'sensitive':
                raise llm_client_base.SensitiveBlockError()
            assert chunk_resp.status.code == 0, f"error: {chunk_resp}"

            chunk = chunk_resp.data
            usage = chunk['usage']
            usage = {
                'prompt_tokens': usage['prompt_tokens'],
                'completion_tokens': usage['completion_tokens'],
            }
            choice0 = chunk['choices'][0]
            role = choice0['role']
            if choice0['finish_reason']:
                finish_reason = choice0['finish_reason']
            result_buffer += choice0['delta']
            if choice0['delta'] and first_token_time is None:
                first_token_time = time.time()

            yield {
                'role': role,
                'delta_content': choice0['delta'],
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

    client = SenseNova_Client()
    model_name = "SenseChat-Turbo"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
