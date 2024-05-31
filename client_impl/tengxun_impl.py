import json
import os
import time

import llm_client_base

from tencentcloud.common import credential
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.hunyuan.v20230901 import hunyuan_client, models
from tencentcloud.common.profile.http_profile import HttpProfile

# config from .env
# TENGXUN_SECRET_ID
# TENGXUN_SECRET_KEY


class Tengxun_Client(llm_client_base.LlmClientBase):
    def __init__(self):
        super().__init__()

        secret_id = os.getenv('TENGXUN_SECRET_ID')
        secret_key = os.getenv('TENGXUN_SECRET_KEY')
        assert secret_id is not None

        self.cred = credential.Credential(secret_id, secret_key)


    async def chat_stream_async(self, model_name, history, temperature, force_calc_token_num):
        start_time = time.time()

        cpf = ClientProfile(httpProfile=HttpProfile(reqTimeout=600))
        # 预先建立连接可以降低访问延迟
        cpf.httpProfile.pre_conn_pool_size = 1
        client = hunyuan_client.HunyuanClient(self.cred, "ap-guangzhou", cpf)

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

        req.Stream = True
        resp = client.ChatCompletions(req)

        result_buffer = ''
        usage = None
        role = None
        finish_reason = None
        first_token_time = None

        for chunk_resp in resp:
            chunk = json.loads(chunk_resp["data"])
            usage = chunk['Usage']
            usage = {
                'prompt_tokens': usage['PromptTokens'],
                'completion_tokens': usage['CompletionTokens'],
            }

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
                }

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'usage': usage,
            'first_token_time': first_token_time - start_time,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    client = Tengxun_Client()
    model_name = "hunyuan-lite"
    history = [{"role": "user", "content": "Hello, how are you?"}]
    temperature = 0.01

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, temperature, force_calc_token_num=True):
            print(chunk)

    asyncio.run(main())
