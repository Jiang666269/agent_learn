#超时控制

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI,APITimeoutError

#加载.env
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

#创建客户端
client = OpenAI(
    api_key = os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout = 0.01, #触发超时异常
    #timeout为单次超时，不代表整个Agent最多运行timeout秒
    max_retries=0  #关闭SDK自动重试
)

#发起请求
try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role":"user",
                "content":"一句话解释什么是timeout"
            }
        ]
    )
    print("请求成功：")
    print(response.choices[0].message.content)
except APITimeoutError:
    print("请求超时：Deepseek未能在规定时间内完成请求")