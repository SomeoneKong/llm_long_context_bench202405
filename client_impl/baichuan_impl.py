

import os

import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# BAICHUAN_API_KEY


class Baichuan_Client(OpenAI_Client):
    def __init__(self):
        api_key = os.getenv('BAICHUAN_API_KEY')

        super().__init__(
            api_base_url="https://api.baichuan-ai.com/v1/",
            api_key=api_key,
        )


if __name__ == '__main__':
    import asyncio
    import os

    client = Baichuan_Client()
    model_name = "Baichuan3-Turbo-128k"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
