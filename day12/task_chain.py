#结合pydantic

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException

#定义数据结构
class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task:str
    priority:Literal["high","medium","low"]
    deadline:str

#创建pydantic输出解析器
#尝试将模型输出解析并验证为task对象
parser = PydanticOutputParser(pydantic_object=Task)

#定义prompt模版
#根据task模型生成结构化输出要求 提供给llm
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "你是一个任务信息提取助手"
        "提取 task、priority、deadline 三个字段"
        "priority 根据紧急程度判断"
        "无法确定的截止时间使用空字符串"
        "只输出 JSON，不要输出其他说明\n"
        "{format_instructions}"
    ),
    ("human","{user_input}")
])

#提前填充格式说明
#partial 预先绑定部分模版变量
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)

#
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

#创建模型
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    temperature = 0,
    timeout = 20,
    max_retries = 0
)

#LCEL
chain = prompt | model | parser

#执行调用链
try:
    result = chain.invoke({
        "user_input":"明天晚上完成数据库作业，非常重要"
    })

    print("任务提取成功")
    print("返回类型：",type(result))
    print("任务:",result.task)
    print("优先级：",result.priority)
    print("截止时间：",result.deadline)

except OutputParserException as error:
    print("模型输出解析或验证失败：")
    print(error)