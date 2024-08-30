

import os
import time

import llm_client_base

from together import AsyncTogether

# config from .env
# TOGETHER_API_KEY


class Together_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self):
        super().__init__()

        api_key = os.getenv('TOGETHER_API_KEY')
        assert api_key is not None

        self.client = AsyncTogether(api_key=api_key)

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']

        start_time = time.time()

        response = await self.client.chat.completions.create(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
        )

        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async for chunk in response:
            # print(chunk)
            choice0 = chunk.choices[0]

            if choice0.delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
            if choice0.finish_reason:
                finish_reason = choice0.finish_reason.name
            if chunk.usage:
                usage = chunk.usage.dict()

            result_buffer += choice0.delta.content
            # print(choice0.delta.content)
            yield {
                'role': choice0.delta.role,
                'delta_content': choice0.delta.content,
                'accumulated_content': result_buffer,
            }

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'usage': usage or {},
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Together_Client()
    model_name = "Qwen/Qwen1.5-72B-Chat"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
