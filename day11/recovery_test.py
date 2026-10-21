#模拟恢复过程

from typing import Literal
from pydantic import BaseModel,ValidationError

#定义任务数据结构
class Task(BaseModel):
    task:str
    priority:Literal["high","medium","low"]
    deadline:str

#模拟第一次LLM返回错误数据
responses = [
    {
        "task":"完成python作业",
        "priority":"urgent",
        "deadline":"明天"
    },
    {
      "task":"完成python作业",
        "priority":"high",
        "deadline":"明天"
    }
]

#最多尝试两次
max_attempts = 2

for attempt in range(max_attempts):
    print(f"\n第{attempt + 1}次验证")

    #模拟获取本轮LLM输出
    task_data = responses[attempt]

    try:
        task = Task.model_validate(task_data)

        print("验证成功")
        print(task.model_dump())

        break
    except ValidationError as error:
        print("验证失败")
        print(error.errors())     #返回一个包含结构化错误信息的列表

        if attempt == max_attempts - 1:
            print("已达到最大尝试次数")