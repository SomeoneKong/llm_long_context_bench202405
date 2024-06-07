

import os
import aiohttp

import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# MOONSHOT_API_KEY


class Moonshot_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('MOONSHOT_API_KEY')
        assert api_key is not None
        self.api_key = api_key

        super().__init__(
            api_base_url="https://api.moonshot.cn/v1",
            api_key=api_key,
        )

    async def count_token(self, model_name, message_list):
        url = "https://api.moonshot.cn/v1/tokenizers/estimate-token-count"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }

        payload = {
            "model": model_name,
            "messages": message_list,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                resp_json = await response.json()
                return resp_json['data']['total_tokens']

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        force_calc_token_num = client_param.get('force_calc_token_num', False)

        async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
            if force_calc_token_num and 'finish_reason' in chunk:
                prompt_token_num = await self.count_token(model_name, history)
                completion_message = {"role": chunk['role'], "content": chunk['accumulated_content']}
                completion_token_num = await self.count_token(model_name, [completion_message])
                chunk['usage'] = {
                    'prompt_tokens': prompt_token_num,
                    'completion_tokens': completion_token_num,
                }

            yield chunk


if __name__ == '__main__':
    import asyncio
    import os

    client = Moonshot_Client()
    model_name = "moonshot-v1-8k"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
