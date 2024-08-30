

import os
import time

import llm_client_base

import cohere

# config from .env
# COHERE_API_KEY


class Cohere_Client(llm_client_base.LlmClientBase):
    support_system_message: bool = True

    def __init__(self):
        super().__init__()

        api_key = os.getenv('COHERE_API_KEY')
        assert api_key is not None

        self.client = cohere.AsyncClient(api_key)

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        model_param = model_param.copy()
        temperature = model_param.pop('temperature')
        max_tokens = model_param.pop('max_tokens', None)

        start_time = time.time()

        message = history[-1]['content']

        send_message_list = []
        for raw_message in history[:-1]:
            role = None
            if raw_message['role'] == 'user':
                role = 'USER'
            elif raw_message['role'] == 'assistant':
                role = 'CHATBOT'
            elif raw_message['role'] == 'system':
                role = 'SYSTEM'
            else:
                raise ValueError(f"Unknown role: {raw_message['role']}")
            send_message_list.append({
                'role':role,
                'message':raw_message['content']
            })

        async_response = self.client.chat_stream(
            message=message,
            model=model_name,
            chat_history=send_message_list,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        role = 'assistant'
        result_buffer = ''
        finish_reason = None
        usage = None
        first_token_time = None

        async for chunk_resp in async_response:
            if chunk_resp.event_type == 'stream-start':
                continue
            elif chunk_resp.event_type == 'text-generation':
                chunk_str = chunk_resp.text
                if first_token_time is None:
                    first_token_time = time.time()

                result_buffer += chunk_str
                # print(choice0.delta.content)
                yield {
                    'role': role,
                    'delta_content': chunk_str,
                    'accumulated_content': result_buffer,
                }
            elif chunk_resp.event_type == 'stream-end':
                response_data = chunk_resp.response
                finish_reason = response_data.finish_reason
                usage = {
                    'prompt_tokens': int(response_data.meta.tokens.input_tokens),
                    'completion_tokens': int(response_data.meta.tokens.output_tokens),
                    # response_data.meta.billed_units.input_tokens,
                    # response_data.meta.billed_units.output_tokens,
                }

        completion_time = time.time()

        yield {
            'role': role,
            'accumulated_content': result_buffer,
            'finish_reason': finish_reason,
            'usage': usage or {},
            'first_token_time': first_token_time - start_time if first_token_time else None,
            'completion_time': completion_time - start_time,
        }


if __name__ == '__main__':
    import asyncio
    import os

    os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890/"
    os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890/"

    client = Cohere_Client()
    # model_name = "command-r-plus-08-2024"
    model_name = "command-r-08-2024"
    history = [
        {"role": "user", "content": "Hello, how are you?"},
        # {"role": "assistant", "content": "I'm fine, thank you."},
        # {"role": "user", "content": "What are you doing?"},
    ]

    model_param = {
        'temperature': 0.01,
    }

    async def main():
        async for chunk in client.chat_stream_async(model_name, history, model_param, client_param={}):
            print(chunk)

    asyncio.run(main())
