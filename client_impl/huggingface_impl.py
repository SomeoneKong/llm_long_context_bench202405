

import os

import llm_client_base

from .openai_impl import OpenAI_Client
from openai import AsyncOpenAI
import huggingface_hub

# config from .env
# HUGGINGFACE_API_KEY


class HuggingfaceEndpoint_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('HUGGINGFACE_API_KEY')
        super().__init__(None, None)

        self.api_key = api_key
        self.client = None

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        # model_name is endpoint name

        endpoint = huggingface_hub.get_inference_endpoint(model_name, token=self.api_key)
        # print(endpoint.url)

        self.client = AsyncOpenAI(
            base_url=endpoint.url + "/v1/",
            api_key=self.api_key,
        )
        async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
            yield chunk

        await self.client.close()


if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = HuggingfaceEndpoint_Client()
    model_name = "qwen2-7b-instruct"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
