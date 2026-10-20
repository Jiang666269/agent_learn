#依赖 "pydantic>=2,<3"

from typing import Literal
from pydantic import BaseModel, ValidationError

# 定义任务数据模型
#根据字段定义对输入数据进行验证
class Task(BaseModel):
    task: str
    priority: Literal["high", "medium", "low"] #Literal限制字段只能取指定值
    deadline: str

#模拟 LLM 返回的 JSON 数据
task_data = {
    "task": "完成 Python 作业",
    "priority": "urgent",
    "deadline": "明天下午五点"
}

#验证数据
try:
    task = Task.model_validate(task_data)

    print("验证成功！")
    print("任务名称：", task.task)
    print("优先级：", task.priority)
    print("截止时间：", task.deadline)

except ValidationError as error:
    print("验证失败！")
    print(error)