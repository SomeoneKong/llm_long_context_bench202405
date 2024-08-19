

import os
import time
import asyncio

import llm_client_base

import sparkai
from sparkai.llm.llm import ChatSparkLLM
from sparkai.core.messages import ChatMessage

# config from .env
# SPARKAI_APP_ID
# SPARKAI_API_KEY
# SPARKAI_API_SECRET

# https://www.xfyun.cn/doc/spark/Web.html


class Xunfei_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self):
        super().__init__()

        self.app_id = os.getenv('SPARKAI_APP_ID')
        self.api_key = os.getenv('SPARKAI_API_KEY')
        self.api_secret = os.getenv('SPARKAI_API_SECRET')
        assert self.api_key is not None

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']

        messages = []
        for message in history:
            messages.append(ChatMessage(
                role=message['role'],
                content=message['content']
            ))

        assert model_name.startswith('spark-')
        model_version = model_name[len('spark-'):]

        if model_version in ['1.5', '2.0', '3.0', '3.5', '4.0']:
            url = f"wss://spark-api.xf-yun.com/v{model_version}/chat"
        elif model_version == '3.0-128k':
            url = f"wss://spark-api.xf-yun.com/chat/pro-128k"
        else:
            assert False, f"Unsupported model version: {model_version}"

        start_time = time.time()
        spark = ChatSparkLLM(
            spark_api_url=url,
            spark_app_id=self.app_id,
            spark_api_key=self.api_key,
            spark_api_secret=self.api_secret,
            # spark_llm_domain="xsstarcoder27binst",
            temperature=temperature,
            streaming=True,
            # max_tokens=1024,
        )

        a = spark.astream(messages)

        await asyncio.sleep(0)

        role = 'assistant'
        finish_reason = 'stop'
        result_buffer = ''
        usage = None
        first_token_time = None

        try:
            async for message in a:
                # print(message)
                delta = message.content
                if 'token_usage' in message.additional_kwargs:
                    usage = message.additional_kwargs['token_usage']
                    del usage['question_tokens']

                result_buffer += delta
                if delta:
                    if first_token_time is None:
                        first_token_time = time.time()
                    yield {
                        'role': role,
                        'delta_content': delta,
                        'accumulated_content': result_buffer,
                    }

                await asyncio.sleep(0)

        except sparkai.errors.SparkAIConnectionError as e:
            if e.error_code in [10013, 10014]:
                raise llm_client_base.SensitiveBlockError() from e

            raise

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'usage': usage,
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Xunfei_Client()
    model_name = "spark-1.5"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
