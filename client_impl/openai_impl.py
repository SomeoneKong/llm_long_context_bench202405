

import os
import time

import llm_client_base

from openai import AsyncOpenAI

# config from .env
# OPENAI_API_KEY
# HTTP_PROXY
# HTTPS_PROXY


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

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param.pop('temperature')
        max_tokens = model_param.pop('max_tokens', None)
        tools = model_param.pop('tools', None)
        json_mode = client_param.get('json_mode', False)

        start_time = time.time()

        system_fingerprint = None
        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None
        real_model = None

        req_args = dict(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
            stream_options={'include_usage': True},
        )
        if json_mode:
            req_args['response_format'] = {"type": "json_object"}
        if max_tokens:
            req_args['max_tokens'] = max_tokens
        if tools:
            req_args['tools'] = tools
        if model_param:
            req_args['extra_body'] = model_param

        async with await self.client.chat.completions.create(**req_args) as response:
            async for chunk in response:
                # print(chunk)
                system_fingerprint = chunk.system_fingerprint
                if chunk.choices:
                    finish_reason = chunk.choices[0].finish_reason
                    delta_info = chunk.choices[0].delta
                    if delta_info:
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
                if chunk.model:
                    real_model = chunk.model


        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'system_fingerprint': system_fingerprint,
            'real_model': real_model,
            'usage': usage or {},
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }

    async def close(self):
        await self.client.close()


if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = OpenAI_Client(api_key=os.getenv('OPENAI_API_KEY'))
    model_name = "gpt-4o-mini"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
        # 'json_mode': True,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
