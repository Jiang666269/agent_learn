#LCEL Chain

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#加载项目根目录的环境变量
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

#定义prompt模版
prompt = ChatPromptTemplate.from_messages([
    ("system","你是一名python教师，请使用中文简介回答。"),
    ("human","请解释什么是{topic},并给出一个简单示例")
])

#创建LLM
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature = 0,
    timeout = 20,
    max_retries = 0
)

#创建输出解析器
parser = StrOutputParser()

#使用LCEL组合组件
#|管道运算符
chain = prompt | model | parser

#执行调用链
result = chain.invoke({
    "topic":"python装饰器"
})

#输出
print("模型回答：")
print(result)
print("\n结果类型：",type(result))