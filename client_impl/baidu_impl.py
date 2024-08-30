

import os
import time

import llm_client_base

# pip install qianfan
import qianfan

# config from .env
# QIANFAN_ACCESS_KEY
# QIANFAN_SECRET_KEY


class Baidu_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self):
        super().__init__()

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']

        system_message_list = [m for m in history if m['role'] == 'system']
        system_prompt = system_message_list[-1]['content'] if system_message_list else None

        message_list = [m for m in history if m['role'] != 'system']

        args = {}
        if 'max_tokens' in model_param:
            args['max_output_tokens'] = min(2047, model_param['max_tokens'])

        start_time = time.time()
        chat_comp = qianfan.ChatCompletion(model=model_name)
        resp = await chat_comp.ado(messages=message_list,
                                   system=system_prompt,
                                   temperature=temperature,
                                   retry_count=0,
                                   stream=True,
                                   **args
                                   )

        result_buffer = ''
        usage = None
        role = 'assistant'
        response_headers = None
        first_token_time = None

        async for chunk_resp in resp:
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

        rate_limit_info = {k[2:]: v for k, v in response_headers.items() if k.startswith('X-Ratelimit')}

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': 0,
            'usage': usage,
            'rate_limit_info': rate_limit_info,
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Baidu_Client()
    model_name = "ERNIE-Speed-8K"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
