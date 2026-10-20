import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

#加载项目目录中的.env
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

#创建deepseek客户端
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

#调用模型，要求返回json
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role":"system",
            "content":(
                "你是一个任务信息提取助手。"
                "请从用户输入中提取 task、priority、deadline 三个字段。"
                "priority 只能是 high、medium、low。"
                "无法确定的 deadline 使用空字符串。"
                "必须输出合法的 JSON 对象。"
            )
        },
        {
            "role":"user",
            "content":"明天下午五点之前完成python作业，非常重要"
        }
    ],
    response_format={"type":"json_object"} #本次响应应采用JSOn对象格式
)

#获取模型返回的JSON字符串
result_text = response.choices[0].message.content

print("模型原始输出：")
print(result_text)

#将JSON字符串转python字典
task_data = json.loads(result_text)

print("\npython解析结果：")
print(task_data)

print("\n任务名称",task_data["task"])
print("任务优先级",task_data["priority"])
print("任务截止时间",task_data["deadline"])