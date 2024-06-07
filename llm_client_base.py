
class LlmClientBase:
    support_system_message: bool

    async def chat_stream_async(self, model_name, history, model_param, client_param):
        raise NotImplementedError()

    async def close(self):
        pass


class SensitiveBlockError(Exception):
    pass
