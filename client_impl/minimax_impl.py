

import os
import time

import llm_client_base

import aiohttp
import json

# config from .env
# MINIMAX_GROUP_ID
# MINIMAX_API_KEY


class Minimax_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = False

    def __init__(self):
        super().__init__()

        self.group_id = os.getenv('MINIMAX_GROUP_ID')
        self.api_key = os.getenv('MINIMAX_API_KEY')
        assert self.api_key is not None

    async def parse_response(self, response):
        pending = None
        async for chunk, _ in response.content.iter_chunks():
            if pending is not None:
                chunk = pending + chunk
            lines = chunk.splitlines()
            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            for line in lines:
                if line.startswith(b'data: '):
                    line = line[6:]
                    if line.startswith(b'{') or line.startswith(b'['):
                        chunk = json.loads(line)
                    else:
                        chunk = line.decode()

                    yield chunk

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        temperature = model_param['temperature']

        url = "https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=" + self.group_id
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        message_list = []
        for message in history:
            if message['role'] == 'user':
                message_list.append({
                    "sender_type": "USER",
                    "sender_name": "小明",
                    "text": message['content'],
                })
            elif message['role'] == 'assistant':
                message_list.append({
                    "sender_type": "BOT",
                    "sender_name": "MM智能助理",
                    "text": message['content'],
                })

        payload = {
            "bot_setting": [
                {
                    "bot_name": "MM智能助理",
                    "content": "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。",
                }
            ],
            "messages": message_list,
            "reply_constraints": {"sender_type": "BOT", "sender_name": "MM智能助理"},
            "model": model_name,
            "stream": True,
            # "tokens_to_generate": 1034,
            "temperature": temperature,
            # "top_p": 0.95,
        }

        start_time = time.time()

        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                async for chunk in self.parse_response(response):
                    choice0 = chunk['choices'][0]
                    # print(choice0)
                    role = choice0['messages'][0]['sender_type']
                    if 'finish_reason' in choice0:
                        finish_reason = choice0['finish_reason']
                        delta_data = ''
                    else:
                        delta_data = choice0['messages'][0]['text']
                        result_buffer += delta_data
                    if 'usage' in chunk:
                        usage = chunk['usage']

                    if role == 'BOT':
                        role = 'assistant'
                    elif role == 'USER':
                        role = 'user'

                    if delta_data:
                        if first_token_time is None:
                            first_token_time = time.time()

                        yield {
                            'role': role,
                            'delta_content': delta_data,
                            'accumulated_content': result_buffer,
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

    client = Minimax_Client()
    model_name = "abab5.5-chat"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
