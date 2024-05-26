# llm_long_context_bench202405

# 简介

本项目包括两部分：
* 各家LLM API SDK 的简单统一包装，未来会随着其他测试项目的需求而不断扩展。
* 包括一个Long context能力和速度的测试方案。

使用方式：
* 在环境变量中配置各家的API key，具体变量名请参考`client_impl`中的具体实现
* 修改并运行 `gen_test_cases.py` 生成测试用例
* 修改`test_main.py`中的测试任务设定，然后运行它来生成测试结果

由于目前各家LLM API的兼容性和请求速率限制问题都较多，所以目前本项目的任务配置和统计都需要人工进行。
