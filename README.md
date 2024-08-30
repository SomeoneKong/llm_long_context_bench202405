# llm_long_context_bench202405

本项目中的LLM client已经接近停止维护，请使用新的独立项目：
[llm_client_wrapper](https://github.com/SomeoneKong/llm_client_wrapper)

# 简介

使用方式：
* 在环境变量中配置各家的API key，具体变量名请参考`client_impl`中的具体实现
* 修改并运行 `gen_test_cases.py` 生成测试用例
* 修改`test_main.py`中的测试任务设定，然后运行它来生成测试结果

由于目前各家LLM API的兼容性和请求速率限制问题都较多，所以目前本项目的任务配置和统计都需要人工进行。


# 测试结果数据

* 每个测试的数据在独立分支上，注意查看本项目的分支。

# 未来可改进点：

* 有些API供应商带有session之间的context prefix缓存，每个请求首次请求时间较长，后续会快。目前的统计中会自动剔除最长的一次，导致第一次的较高延迟没有被纳入。
  需要考虑设计能评估带有缓存方案的性能。
