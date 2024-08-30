

import os

import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# MINIMAX_API_KEY


class Minimax_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('MINIMAX_API_KEY')

        super().__init__(
            api_base_url="https://api.minimax.chat/v1/",
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        temp_model_param = model_param.copy()
        if 'max_tokens' not in temp_model_param:
            temp_model_param['max_tokens'] = 2048  # 官方默认值为256，太短

        async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
            if 'finish_reason' in chunk:
                assert chunk.get('first_token_time', None) is not None, f"minimax return empty"

            yield chunk


if __name__ == '__main__':
    import asyncio
    import os

    client = Minimax_Client()
    model_name = "abab6.5s-chat"

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        history = [{"role": "user", "content": "Hello, how are you?"}]
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
