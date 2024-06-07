

import os

import llm_client_base
import openai
from .openai_impl import OpenAI_Client

# config from .env
# DEEPSEEK_API_KEY


class DeepSeek_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('DEEPSEEK_API_KEY')

        super().__init__(
            api_base_url="https://api.deepseek.com/v1",
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        try:
            async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
                yield chunk
        except openai.BadRequestError as e:
            if 'Content Exists Risk' in e.message:
                raise llm_client_base.SensitiveBlockError() from e

            raise


if __name__ == '__main__':
    import asyncio
    import os

    client = DeepSeek_Client()
    model_name = "deepseek-chat"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
