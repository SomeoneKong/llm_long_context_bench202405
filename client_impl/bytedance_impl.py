

import os

import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# VOLC_API_KEY


class ByteDance_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('VOLC_API_KEY')

        super().__init__(
            api_base_url="https://ark.cn-beijing.volces.com/api/v3/",
            api_key=api_key,
        )


if __name__ == '__main__':
    import asyncio
    import os

    client = ByteDance_Client()
    model_name = "ep-xxxxxx"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
