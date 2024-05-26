

import os

import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# YI_API_KEY


class Yi_Client(OpenAI_Client):
    def __init__(self):
        api_key = os.getenv('YI_API_KEY')

        super().__init__(
            api_base_url="https://api.lingyiwanwu.com/v1",
            api_key=api_key,
        )


if __name__ == '__main__':
    import asyncio
    import os

    client = Yi_Client()
    model_name = "yi-spark"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
