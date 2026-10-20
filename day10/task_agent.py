#整体与llm相同，修改了用户输入方式

import os
import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

#定义数据结构
class Task(BaseModel):
    task: str
    priority: Literal["high", "medium", "low"]
    deadline: str

#加载项目根目录的 .env
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

#创建 DeepSeek 客户端
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

#用户的自然语言输入
user_input = input("请输入你的任务：").strip()


#让 LLM 提取任务信息
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": (
                "你是任务信息提取助手。"
                "从用户输入中提取 task、priority、deadline 三个字段。"
                "priority 只能是 high、medium 或 low。"
                "如果无法确定截止时间，deadline 使用空字符串。"
                "必须返回包含这三个字段的合法 JSON 对象。"
            )
        },
        {
            "role": "user",
            "content": user_input
        }
    ],
    response_format={"type": "json_object"}
)

#取得 JSON 字符串
result_text = response.choices[0].message.content

print("LLM 原始输出：")
print(result_text)

#解析并验证
try:
    task_dict = json.loads(result_text)

    task = Task.model_validate(task_dict)

    print("\nPydantic 验证成功！")
    print("任务：", task.task)
    print("优先级：", task.priority)
    print("截止时间：", task.deadline)

except json.JSONDecodeError as error:   #JSON解析错误验证
    print("\nJSON 解析失败：")
    print(error)

except ValidationError as error:        #pydantic验证
    print("\nPydantic 验证失败：")
    print(error)