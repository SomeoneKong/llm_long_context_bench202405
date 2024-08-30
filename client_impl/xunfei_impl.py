

import os
import time

import llm_client_base
import openai
from .openai_impl import OpenAI_Client

# config from .env
# SPARKAI_HTTP_API_KEY


class Xunfei_Client(OpenAI_Client):
    # https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html

    support_system_message: bool = True

    def __init__(self):
        self.api_key = os.getenv('SPARKAI_HTTP_API_KEY')
        assert self.api_key is not None

        super().__init__(
            api_base_url='https://spark-api-open.xf-yun.com/v1',
            api_key=self.api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        assert model_name.startswith('spark-')
        model_version = model_name[len('spark-'):]
        assert model_version in model_version in ['lite', 'pro', 'pro-128k', 'max', '4.0']

        spark_model_name_dict = {
            'lite': 'general',
            'pro': 'generalv3',
            'pro-128k': 'pro-128k',
            'max': 'generalv3.5',
            '4.0': '4.0Ultra',
        }
        spark_model_name = spark_model_name_dict[model_version]

        async for chunk in super().chat_stream_async(spark_model_name, history, model_param, client_param):
            yield chunk

if __name__ == '__main__':
    import asyncio
    import os

    client = Xunfei_Client()
    model_name = "spark-lite"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
