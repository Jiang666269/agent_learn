import os
import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv() #加载.env环境变量

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def calculator(a: int, b: int):
    return a + b


tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算两个整数的加法",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
    }
]


messages = [
    {
        "role": "user",
        "content": "123456 + 789012 等于多少？"
    }
]


response = client.chat.completions.create(
    model="deepseek-flash",
    messages=messages,
    tools=tools
)

message = response.choices[0].message

tool_call = message.tool_calls[0]

arguments = json.loads(
    tool_call.function.arguments
)

result = calculator(
    arguments["a"],
    arguments["b"]
)

print(result)