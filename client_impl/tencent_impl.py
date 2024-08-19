import json
import os
import time
import asyncio

import llm_client_base

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
from tencentcloud.common.profile.http_profile import HttpProfile

# config from .env
# TENCENT_SECRET_ID
# TENCENT_SECRET_KEY


class Tencent_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self):
        super().__init__()

        secret_id = os.getenv('TENCENT_SECRET_ID')
        secret_key = os.getenv('TENCENT_SECRET_KEY')
        assert secret_id is not None

        self.cred = credential.Credential(secret_id, secret_key)

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        enable_search = model_param.get('enable_search', False)

        # 通过插入sleep来把同步API拆分成接近异步的API
        await asyncio.sleep(0)

        start_time = time.time()

        cpf = ClientProfile(httpProfile=HttpProfile(reqTimeout=600))
        client = hunyuan_client.HunyuanClient(self.cred, "ap-guangzhou", cpf)

        await asyncio.sleep(0)

        req = models.ChatCompletionsRequest()
        req.Model = model_name
        req.Temperature = temperature

        message = []
        for m in history:
            msg = models.Message()
            msg.Role = m['role']
            msg.Content = m['content']
            message.append(msg)
        req.Messages = message

        if enable_search:
            req.SearchInfo = True
            req.Citation = True
            req.EnableEnhancement = True

        req.Stream = True
        resp = client.ChatCompletions(req)

        await asyncio.sleep(0)

        result_buffer = ''
        usage = None
        role = None
        finish_reason = None
        search_results = None
        first_token_time = None

        for chunk_resp in resp:
            chunk = json.loads(chunk_resp["data"])
            usage = chunk['Usage']
            usage = {
                'prompt_tokens': usage['PromptTokens'],
                'completion_tokens': usage['CompletionTokens'],
            }

            if 'SearchInfo' in chunk:
                search_results = chunk['SearchInfo']['SearchResults']

            choice0 = chunk['Choices'][0]

            if choice0['FinishReason']:
                finish_reason = choice0['FinishReason']

            if 'Delta' in choice0:
                role = choice0['Delta']['Role']
                result_buffer += choice0['Delta']['Content']
                if choice0['Delta']['Content'] and first_token_time is None:
                    first_token_time = time.time()

                yield {
                    'role': role,
                    'delta_content': choice0['Delta']['Content'],
                    'accumulated_content': result_buffer,
                    'usage': usage,
                    'search_results': search_results,
                }

            await asyncio.sleep(0)

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'search_results': search_results,
            'usage': usage,
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Tencent_Client()
    model_name = "hunyuan-lite"
    history = [{"role": "user", "content": "Hello, how are you?"}]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
