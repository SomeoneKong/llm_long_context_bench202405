

import os
import time

from openai import AsyncOpenAI


class LlmClientBase:
    support_system_message: bool

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        raise NotImplementedError()

    async def close(self):
        pass

class OpenAI_Client(LlmClientBase):
    support_system_message: bool = True

    def __init__(self,
                 api_base_url=None,
                 api_key=None,
                 ):
        super().__init__()
        self.client = AsyncOpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        temperature = model_param['temperature']
        force_calc_token_num = client_param.get('force_calc_token_num', False)

        start_time = time.time()

        system_fingerprint = None
        role = None
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async with await self.client.chat.completions.create(
            model=model_name,
            messages=history,
            temperature=temperature,
            stream=True,
            stream_options={'include_usage': True},
        ) as response:

            async for chunk in response:
                system_fingerprint = chunk.system_fingerprint
                if chunk.choices:
                    finish_reason = chunk.choices[0].finish_reason
                    delta_info = chunk.choices[0].delta
                    if delta_info.role:
                        role = delta_info.role
                    if delta_info.content:
                        result_buffer += delta_info.content

                        if first_token_time is None:
                            first_token_time = time.time()

                        yield {
                            'role': role,
                            'delta_content': delta_info.content,
                            'accumulated_content': result_buffer,
                        }
                if chunk.usage:
                    usage = chunk.usage.dict()


        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'system_fingerprint': system_fingerprint,
            'usage': usage or {},
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }

    async def close(self):
        await self.client.close()



class Vllm_Client(OpenAI_Client):
    support_system_message: bool = True

    def __init__(self):
        super().__init__(
            api_base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )

import asyncio
import time


async def run_test(client_factory, model_name, prompt):
    client = client_factory()
    history = [{"role": "user", "content": prompt}]
    temperature = 0.8

    model_param = {
        'temperature': temperature,
    }
    client_param = {
        'force_calc_token_num': True,
    }

    result = ''
    usage = None
    async for chunk in client.chat_stream_async(model_name, history, model_param, client_param):
        result = chunk['accumulated_content']
        if 'usage' in chunk:
            usage = chunk['usage']

        if 'delta_content' in chunk:
            print(chunk['delta_content'], end='', flush=True)
        else:
            print()

    if result == '':
        print(f'finish_reason: {chunk["finish_reason"]}')

    await client.close()

    ret = {
        'result': result,
        'usage': usage,
        'finish_reason': chunk['finish_reason'],
        'first_token_time': chunk['first_token_time'],
        'total_time': chunk['completion_time'],
    }
    if usage and 'completion_tokens' in usage and chunk['completion_time'] > chunk['first_token_time']:
        ret['token_speed'] = usage['completion_tokens'] / (chunk['completion_time'] - chunk['first_token_time'])
    return ret


def test_128k():

    gap_time = 0

    client_factory, model_name, gap_time = Vllm_Client, "glm-4-9b-chat", 0

    test_file_list = [
        'test_case3v2_128k_sample1.txt',
        'test_case3v2_128k_sample2.txt',
        'test_case3v2_128k_sample3.txt',

        'test_case4v2_128k_sample1.txt',
        'test_case4v2_128k_sample2.txt',
        'test_case4v2_128k_sample3.txt',
    ]

    print(f'model_name: {model_name}')
    for test_file in test_file_list:
        print(f'=================== {test_file} ===============')
        prompt = open(test_file, 'r', encoding='utf8').read()

        output_sample_num = 5
        success_num = 0
        first_token_time_list = []
        token_speed_list = []
        prompt_token = None
        for i in range(output_sample_num):
            print(f'-------------- {test_file} {i}--------------')
            start_time = time.time()
            result = asyncio.run(run_test(client_factory, model_name, prompt))
            print('-------------------------------')

            if 'prompt_tokens' in result['usage']:
                prompt_token = result['usage']['prompt_tokens']

            success = '1350' in result['result']
            if success:
                success_num += 1

            print(f'usage: {result["usage"]}')
            if 'rate_limit_info' in result:
                print(f'rate_limit_info: {result["rate_limit_info"]}')
            print(f'first token time: {result["first_token_time"]}')
            first_token_time_list.append(result["first_token_time"])
            if 'token_speed' in result:
                print(f'token speed: {result["token_speed"]}')
                token_speed_list.append(result["token_speed"])

            if gap_time > 0:
                sleep_time = max(3, gap_time - (time.time() - start_time))
                time.sleep(sleep_time)

        print(f'===================STAT {test_file} ===============')
        print(f'prompt token: {prompt_token}')
        print(f'success rate: {success_num}/{output_sample_num}')

        stat_first_token_time_list = sorted(first_token_time_list)[:-1]
        avg_first_token_time = sum(stat_first_token_time_list) / len(stat_first_token_time_list)
        print(f'avg_first_token_time: {avg_first_token_time}, {first_token_time_list}')

        if token_speed_list:
            stat_token_speed_list = sorted(token_speed_list)[1:]
            avg_token_speed = sum(stat_token_speed_list) / len(stat_token_speed_list)
            print(f'avg_token_speed: {avg_token_speed}, {token_speed_list}')


def test_32k():
    gap_time = 0

    client_factory, model_name, gap_time = Vllm_Client, "glm-4-9b-chat", 0

    test_file_list = [
        'test_case3v2_32k_sample1.txt',
        'test_case3v2_32k_sample2.txt',
        'test_case3v2_32k_sample3.txt',

        'test_case4v2_32k_sample1.txt',
        'test_case4v2_32k_sample2.txt',
        'test_case4v2_32k_sample3.txt',
    ]

    print(f'model_name: {model_name}')
    for test_file in test_file_list:
        print(f'=================== {test_file} ===============')
        prompt = open(test_file, 'r', encoding='utf8').read()

        output_sample_num = 5
        success_num = 0
        first_token_time_list = []
        token_speed_list = []
        prompt_token = None
        for i in range(output_sample_num):
            print(f'-------------- {test_file} {i}--------------')
            start_time = time.time()
            result = asyncio.run(run_test(client_factory, model_name, prompt))
            print('-------------------------------')

            if 'prompt_tokens' in result['usage']:
                prompt_token = result['usage']['prompt_tokens']

            success = '1350' in result['result']
            if success:
                success_num += 1

            print(f'usage: {result["usage"]}')
            if 'rate_limit_info' in result:
                print(f'rate_limit_info: {result["rate_limit_info"]}')
            print(f'first token time: {result["first_token_time"]}')
            first_token_time_list.append(result["first_token_time"])
            if 'token_speed' in result:
                print(f'token speed: {result["token_speed"]}')
                token_speed_list.append(result["token_speed"])

            if gap_time > 0:
                sleep_time = max(3, gap_time - (time.time() - start_time))
                time.sleep(sleep_time)

        print(f'===================STAT {test_file} ===============')
        print(f'prompt token: {prompt_token}')
        print(f'success rate: {success_num}/{output_sample_num}')

        stat_first_token_time_list = sorted(first_token_time_list)[:-1]
        avg_first_token_time = sum(stat_first_token_time_list) / len(stat_first_token_time_list)
        print(f'avg_first_token_time: {avg_first_token_time}, {first_token_time_list}')

        if token_speed_list:
            stat_token_speed_list = sorted(token_speed_list)[1:]
            avg_token_speed = sum(stat_token_speed_list) / len(stat_token_speed_list)
            print(f'avg_token_speed: {avg_token_speed}, {token_speed_list}')


if __name__ == '__main__':
    import os
    # test_128k()

    test_32k()
