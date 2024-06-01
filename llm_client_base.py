
class LlmClientBase:
    support_system_message: bool

    async def chat_stream_async(self, model_name, history, temperature, force_calc_token_num):
        raise NotImplementedError()

    async def close(self):
        pass


class SensitiveBlockError(Exception):
    pass
