

import os
import time

import llm_client_base

import reka

# config from .env
# REKA_API_KEY


class Reka_Client(llm_client_base.LlmClientBase):
    def __init__(self):
        super().__init__()

        api_key = os.getenv('REKA_API_KEY')
        assert api_key is not None

        reka.API_KEY = api_key

    async def chat_stream_async(self, model_name, history, temperature, force_calc_token_num):

        message_list = []
        for message in history[:-1]:
            if message['role'] == 'user':
                message_list.append({
                    "type": "human",
                    "text": message['content'],
                })
            elif message['role'] == 'assistant':
                message_list.append({
                    "type": "model",
                    "text": message['content'],
                })

        last_message = history[-1]['content']

        start_time = time.time()

        response = reka.chat(last_message,
                             conversation_history=message_list,
                             model_name=model_name,
                             temperature=temperature
                             )

        role = 'assistant'
        finish_reason = response['finish_reason']
        usage = response['metadata']
        usage = {
            'prompt_tokens': usage['input_tokens'],
            'completion_tokens': usage['generated_tokens'],
        }
        result = response['text']

        completion_time = time.time()

        yield {
            'role': role,
            'delta_content': result,
            'accumulated_content': result,
            'usage': usage,
        }

        yield {
            'role': role,
            'accumulated_content': result,
            'finish_reason': finish_reason,
            'usage': usage,
            'first_token_time': completion_time - start_time,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Reka_Client()
    model_name = "reka-core"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
