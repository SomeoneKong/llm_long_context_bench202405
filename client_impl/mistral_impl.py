

import os
import time

import llm_client_base

from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage

# config from .env
# MISTRAL_API_KEY


class Mistral_Client(llm_client_base.LlmClientBase):
    def __init__(self):
        super().__init__()

        api_key = os.getenv('MISTRAL_API_KEY')
        assert api_key is not None

        self.client = MistralAsyncClient(api_key=api_key)

    async def chat_stream_async(self, model_name, history, temperature, force_calc_token_num):

        message_list = [
            ChatMessage(role=message['role'], content=message['content'])
            for message in history
        ]

        start_time = time.time()

        async_response = self.client.chat_stream(model=model_name, messages=message_list)

        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async for chunk in async_response:
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
            'first_token_time': first_token_time - start_time,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Mistral_Client()
    model_name = "mistral-small-latest"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
