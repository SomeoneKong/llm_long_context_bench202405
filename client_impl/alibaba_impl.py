

import os

import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# DASHSCOPE_API_KEY


class Alibaba_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('DASHSCOPE_API_KEY')

        super().__init__(
            api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        if model_name in ['qwen-turbo', 'qwen-plus', 'qwen-max'] and 'max_tokens' in model_param:
            model_param['max_tokens'] = min(2000, model_param['max_tokens'])

        async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
            yield chunk


if __name__ == '__main__':
    import asyncio
    import os

    client = Alibaba_Client()
    model_name = "qwen-turbo"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
