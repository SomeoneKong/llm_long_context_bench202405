import json
import os
import time
import hashlib, hmac
import datetime
import aiohttp

import llm_client_base

# config from .env
# TENCENT_SECRET_ID
# TENCENT_SECRET_KEY


class Tencent_Client(llm_client_base.LlmClientBase):
    # https://cloud.tencent.com/document/product/1729/105701

    support_system_message: bool = True

    def __init__(self):
        super().__init__()

        secret_id = os.getenv('TENCENT_SECRET_ID')
        secret_key = os.getenv('TENCENT_SECRET_KEY')
        assert secret_id is not None

        self.secret_id = secret_id
        self.secret_key = secret_key

    def _gen_tencent_auth_header(
        self,
        input_headers,
        other_infos,
        payload: str,
        signed_header_list
    ):
        secret_id = self.secret_id
        secret_key = self.secret_key

        output_headers = {
            **input_headers,
        }

        timestamp = output_headers.get('X-TC-Timestamp')
        if timestamp is None:
            timestamp = int(time.time())
            output_headers['X-TC-Timestamp'] = str(timestamp)
        date = datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")

        host = other_infos.get('Host')
        service = host.split('.')[0]

        # ************* 步骤 1：拼接规范请求串 *************
        http_request_method = "POST"
        canonical_uri = "/"
        canonical_querystring = ""

        info_dict = {}
        for k, v in input_headers.items():
            info_dict[k.lower()] = v
        for k, v in other_infos.items():
            info_dict[k.lower()] = v

        signed_header_list = [h.lower() for h in signed_header_list]
        signed_headers = ';'.join(signed_header_list)

        canonical_header_list = []
        for k in signed_header_list:
            canonical_header_list.append(k + ":" + info_dict[k].lower())
        canonical_headers = '\n'.join(canonical_header_list) + '\n'

        hashed_request_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        canonical_request = (http_request_method + "\n" +
                             canonical_uri + "\n" +
                             canonical_querystring + "\n" +
                             canonical_headers + "\n" +
                             signed_headers + "\n" +
                             hashed_request_payload)
        # print(canonical_request)

        # ************* 步骤 2：拼接待签名字符串 *************
        algorithm = "TC3-HMAC-SHA256"
        credential_scope = date + "/" + service + "/" + "tc3_request"
        hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
        string_to_sign = (algorithm + "\n" +
                          str(timestamp) + "\n" +
                          credential_scope + "\n" +
                          hashed_canonical_request)
        # print(string_to_sign)


        # ************* 步骤 3：计算签名 *************
        # 计算签名摘要函数
        def sign(key, msg):
            return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
        secret_date = sign(("TC3" + secret_key).encode("utf-8"), date)
        secret_service = sign(secret_date, service)
        secret_signing = sign(secret_service, "tc3_request")
        signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        # print(signature)

        # ************* 步骤 4：拼接 Authorization *************
        authorization = (algorithm + " " +
                         "Credential=" + secret_id + "/" + credential_scope + ", " +
                         "SignedHeaders=" + signed_headers + ", " +
                         "Signature=" + signature)
        # print(authorization)

        output_headers['Authorization'] = authorization

        return output_headers

    async def _parse_response(self, response):
        pending = None
        async for chunk, _ in response.content.iter_chunks():
            if pending is not None:
                chunk = pending + chunk
            lines = chunk.splitlines()
            if lines and lines[-1] and chunk and lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None

            for line in lines:
                if line.startswith(b'data: '):
                    line = line[6:]
                    if line.startswith(b'{') or line.startswith(b'['):
                        chunk = json.loads(line)
                    else:
                        chunk = line.decode()

                    yield chunk

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param['temperature']
        enable_search = model_param.get('enable_search', False)

        start_time = time.time()

        url = "https://hunyuan.tencentcloudapi.com"
        headers = {
            "Content-Type": "application/json",
            "X-TC-Version": "2023-09-01",
            "X-TC-Action": "ChatCompletions",
        }

        other_infos = {
            "Host": "hunyuan.tencentcloudapi.com",
        }

        signed_header_list = ['Content-Type', 'Host']

        messages = []
        for m in history:
            msg = {
                "Role": m['role'],
                "Content": m['content']
            }
            messages.append(msg)

        payload = {
            "Model": model_name,
            "Temperature": temperature,
            "Stream": True,
            "Messages": messages,
        }

        if enable_search:
            payload["SearchInfo"] = True
            payload["Citation"] = True
            payload["EnableEnhancement"] = True

        payload_data = json.dumps(payload)
        headers = self._gen_tencent_auth_header(
            headers,
            other_infos,
            payload_data,
            signed_header_list
        )


        result_buffer = ''
        usage = None
        role = None
        finish_reason = None
        search_results = None
        first_token_time = None

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                async for chunk in self._parse_response(response):
                    if 'Response' in chunk:
                        assert False, f"error: {chunk['Response']}"

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
