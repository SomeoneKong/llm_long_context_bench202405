
import os

import llm_client_base
from .openai_impl import OpenAI_Client

# config from .env
# ZHIPU_API_KEY


class Zhipu_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('ZHIPU_API_KEY')
        assert api_key is not None

        super().__init__(
            api_base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key=api_key,
        )


if __name__ == '__main__':
    import asyncio
    import os

    client = Zhipu_Client()
    model_name = "glm-3-turbo"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
