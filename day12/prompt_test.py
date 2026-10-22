#Prompt 模版生成机制

from langchain_core.prompts import ChatPromptTemplate

#定义prompt模版
prompt = ChatPromptTemplate.from_messages([
    ("system","你是一名{language}教师，请简介回答"),
    ("human","请解释什么是{topic}")
])

#向模版传入变量
#invoke根据传入变量生成格式化后的prompt
messages = prompt.invoke({
    "language":"rust",
    "topic":"所有权机制"
})

#输出
for message in messages.messages:
    print(f"角色：{message.type}")
    print(f"内容：{message.content}")
    print("-" * 30)