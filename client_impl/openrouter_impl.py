

import os

from .openai_impl import OpenAI_Client

# config from .env
# OPENROUTER_API_KEY

# https://openrouter.ai/docs/quick-start
# 模型列表 https://openrouter.ai/models


class OpenRouter_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('OPENROUTER_API_KEY')
        assert api_key is not None
        self.api_key = api_key

        super().__init__(
            api_base_url="https://openrouter.ai/api/v1/",
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param_temp = model_param.copy()
        model_name_temp = model_name
        if '|' in model_name:
            chunks = model_name.split('|')
            model_name_temp = chunks[0]
            provider = chunks[1]
            model_param_temp['provider'] = {'order': [provider]}

        async for chunk in super().chat_stream_async(model_name_temp, history, model_param_temp, client_param):
            yield chunk


if __name__ == '__main__':
    import asyncio
    import os

    client = OpenRouter_Client()
    model_name = "anthropic/claude-3.5-sonnet"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
