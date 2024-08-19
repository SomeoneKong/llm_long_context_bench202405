

import os
import time

import llm_client_base

from reka import ChatMessage
from reka.client import AsyncReka

# config from .env
# REKA_API_KEY


class Reka_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = False

    def __init__(self):
        super().__init__()

        api_key = os.getenv('REKA_API_KEY')
        assert api_key is not None

        self.client = AsyncReka(api_key=api_key)

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']

        message_list = []
        for message in history:
            message_list.append(ChatMessage(content=message['content'], role=message['role']))

        start_time = time.time()

        response = self.client.chat.create_stream(
            messages=message_list,
            model=model_name,
            temperature=temperature
            )

        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async for resp in response:
            if resp.usage:
                usage = {
                    'prompt_tokens': resp.usage.input_tokens,
                    'completion_tokens': resp.usage.output_tokens,
                }

            if resp.responses:
                choice0 = resp.responses[0].chunk

                delta = choice0.content[len(result_buffer):]
                result_buffer = choice0.content
                role = choice0.role
                finish_reason = resp.responses[0].finish_reason

                if first_token_time is None:
                    first_token_time = time.time()

                if delta:
                    yield {
                        'role': role,
                        'delta_content': delta,
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

    client = Reka_Client()
    model_name = "reka-core"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
