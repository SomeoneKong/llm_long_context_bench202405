

import os
import openai
import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# STEPFUN_API_KEY


class StepFun_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('STEPFUN_API_KEY')

        super().__init__(
            api_base_url="https://api.stepfun.com/v1",
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        try:
            async for chunk in super().chat_stream_async(model_name, history, model_param, client_param):
                yield chunk
        except openai.APIStatusError as e:
            if 'censorship_blocked' in e.message:
                raise llm_client_base.SensitiveBlockError() from e

            raise


if __name__ == '__main__':
    import asyncio
    import os

    client = StepFun_Client()
    model_name = "step-1-32k"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
