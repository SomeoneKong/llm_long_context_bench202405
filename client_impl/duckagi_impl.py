

import os
import aiohttp

import llm_client_base

from .openai_impl import OpenAI_Client

# config from .env
# DUCKAGI_API_KEY

# https://api.duckagi.com/
# 模型列表 https://api.duckagi.com/pricing


class DuckAgi_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        api_key = os.getenv('DUCKAGI_API_KEY')
        assert api_key is not None
        self.api_key = api_key

        super().__init__(
            api_base_url="https://api.duckagi.com/v1/",
            api_key=api_key,
        )

    async def count_token(self, model_name, message_list):
        if model_name.startswith('claude-'):
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            assert api_key is not None

            client = anthropic.AsyncAnthropic(
                api_key=api_key
            )

            token_num = 0
            for message in message_list:
                token_num += await client.count_tokens(message['content'])

            await client.close()
            return token_num

    async def chat_stream_async(self, model_name, history, temperature, force_calc_token_num):
        async for chunk in super().chat_stream_async(model_name, history, temperature, force_calc_token_num):
            if force_calc_token_num and 'finish_reason' in chunk:
                prompt_token_num = await self.count_token(model_name, history)
                if prompt_token_num:
                    completion_message = {"role": chunk['role'], "content": chunk['accumulated_content']}
                    completion_token_num = await self.count_token(model_name, [completion_message])
                    chunk['usage'] = {
                        'prompt_tokens': prompt_token_num,
                        'completion_tokens': completion_token_num,
                    }

            yield chunk


if __name__ == '__main__':
    import asyncio
    import os

    client = DuckAgi_Client()
    model_name = "claude-3-haiku-20240307"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
