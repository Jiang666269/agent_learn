#依赖 langchain-deepseek python-dotenv

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

#加载env
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

#创建Langchain的deepseek模型对象
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature=0,
    timeout=20,
    max_retries=0
)

#调用模型
#invoke 是统一调用接口
#返回AIMessage对象
response = model.invoke([
    ("system","你是一名python教师，请简洁回答"),
    ("human","什么是python装饰器")
])

print(response.content)