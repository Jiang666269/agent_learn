#Agent Loop(自动循环),引入多个tool，与用户交互，多轮对话，控制Context长度
#改造day1

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def calculator(a: int, b: int):
    return a + b


def multiply(a: int, b: int):
    return a * b

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算两个整数的加法",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer"
                    },
                    "b": {
                        "type": "integer"
                    }
                },
                "required": ["a", "b"]
            }
        }
    },

    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个整数的乘法",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer"
                    },
                    "b": {
                        "type": "integer"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]

tool_map = {
    "calculator": calculator,
    "multiply": multiply
}#工具注册表

messages = [
    {
        "role":"system",
        "content":"你是一个计算Agent，需要计算时使用提供的工具，回答时不要使用Markdown加粗"
    }
]
#最多保留10条对话
MAX_HISTORY = 10

#裁剪Context

def trim_messages():
    global messages

    system_message = messages[0]

    history = []

    for msg in messages[1:]:
        # 保留用户消息
        if msg.get("role") == "user":
            history.append(msg)

        # 只保留正常的 AI 最终回答
        elif (
                msg.get("role") == "assistant"
                and msg.get("content")
                and not msg.get("tool_calls")
        ):
            history.append(msg)

        # 只保留最近 MAX_HISTORY 条
    history = history[-MAX_HISTORY:]

    messages = [system_message] + history
#用户输入
def run_agent(user_input):

    messages.append({
        "role": "user",
        "content": user_input
        }#保持聊天的连续性
    )

    max_steps = 10  # 避免模型错误判断导致无限循环

    for step in range(max_steps):

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message

        if not message.tool_calls:

            messages.append({
                "role":"assistant",
                "content":message.content
            })

            trim_messages()

            return message.content

        messages.append(
            message.model_dump(exclude_none=True)
        )

        for tool_call in message.tool_calls:

            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )#json转为字典

            tool_function = tool_map[tool_name]

            result = tool_function(**arguments)  # **将python字典拆成函数的关键字参数，等价于tool_function(a=12,b=5)

            messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
            })

    return "Agent执行次数超过限制"

while True:

    question = input("你：")

    if question == "退出": #可优化
        break

    answer = run_agent(question)

    print("AI:",answer)