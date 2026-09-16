#day3 长期记忆Memory，读取，保存，更新，删除，Memory语义去重、限制key、冲突覆盖，同时保存多条Memory，Memory Retrieval，Structured Output +JSON解析保护

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
#读取memory的函数
def load_memory():
    try:
        with open("memory.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

#保存memory的函数
def save_memory(memory):
    with open("memory.json", "w", encoding="utf-8") as file:
        json.dump(
            memory,
            file,
            ensure_ascii=False,#保存为自然语言写进JSON
            indent=4 #格式化JSON
        )

#将memory转换成模型能看到的文本
def build_system_prompt():
    memory = load_memory()

    memory_text = json.dumps(
        memory,
        ensure_ascii=False,
        indent=2
    )
#让agent决定什么时候写入memory
    return f"""
你是一个计算与学习 Agent。

你可以调用工具完成任务。

只有当用户提供长期稳定的信息、长期偏好、学习方向、未来对回答有帮助的信息时，
才应该调用 remember 工具保存。

如果用户一次提供多条独立的长期信息
可以分别调用remember工具保存每一条信息

不要把临时事件、一次性问题等内容保存为长期记忆。

当前用户长期 Memory：

{memory_text}

回答时不要使用 Markdown 加粗。
"""

#删除长期记忆
"""
def forget(key:str):
    memory = load_memory()

    if key in memory:
        del memory[key]
        save_memory(memory)
        return f"已删除记忆：{key}"

    return f"没有找到记忆:{key}"
"""

def calculator(a: int, b: int):
    return a + b

def multiply(a: int, b: int):
    return a * b

#限制key，示例
'''
ALLOWED_MEMORY_KEYS = {
    "code_language",
    "answer_language",
    "learning_topic",
    "answer_style"
}'''

#检索Memory中有关此次问题的
def retrieve_memory(user_input):
    memory = load_memory()

    if not memory:
        return []

    prompt = f"""
用户当前问题：
{user_input}
已有长期记忆：
{json.dumps(memory, ensure_ascii=False, indent=2)}
选择与当前问题有帮助的长期记忆。
只返回 JSON：
{{
    "indices": [相关记忆的下标]
}}
如果没有相关记忆：
{{
    "indices": []
}}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = parse_json_response(
        response.choices[0].message.content
    )

    if result is None or not isinstance(result,dict):
        return []

    indices = result.get("indices",[])

    return [
        memory[index]
        for index in indices
        if isinstance(index,int) and  0 <= index < len(memory)
    ]

#安全解析函数
def parse_json_response(text):

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1])

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        return None
#模型返回结构检查
def validate_memory_result(result):

    if not isinstance(result, dict):
        return False

    action = result.get("action")
    index = result.get("index")

    if action not in [
        "duplicate",
        "new",
        "update"
    ]:
        return False

    if action == "new":
        return index is None

    if action in ["duplicate", "update"]:
        return isinstance(index, int)

    return False

#判断function
def classify_memory(memory, new_memory):

    prompt = f"""
已有长期记忆：
{json.dumps(memory, ensure_ascii=False)}
准备保存的新记忆：
{json.dumps(new_memory, ensure_ascii=False)}
请判断新记忆属于以下哪一种：
duplicate：与已有记忆表达基本相同
new：与已有记忆无明显冲突，是新的长期信息
update：与已有的某条记忆属于同一信息维度，但新内容发生了变化，应更新旧记忆
请只返回JSON。
如果是 new：
{{
    "action":"new",
    "index":null
}}

如果是 duplicate：
{{
    "action": "duplicate",
    "index": 0
}}

如果是 update:
{{
    "action": "update",
    "index": 0
}}
index 应填写实际对应旧记忆的列表下标，0只是格式示例
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result_text = response.choices[0].message.content.strip().lower()
    #防止模型输出格式有问题
    result = parse_json_response(result_text)

    if result is None:
        return{
            "action":"error",
            "index":None
        }
    if not validate_memory_result(result):
        return{
            "action":"error",
            "index":None
        }
    return result

#保存memory的工具,Memory Entry
def remember(memory_type: str, content: str):
    memory = load_memory()

    new_memory = {
        "type":memory_type,
        "content":content
    }
    #精准去重
    if new_memory in memory:
        return "该记忆已经存在"

    result = classify_memory(memory,new_memory)

    action = result["action"]
    index = result["index"]
    #语义去重
    if action == "duplicate":
        return "存在语义相同的记忆"

    elif action == "new":
        memory.append(new_memory)

        save_memory(memory)

        return "新记忆保存成功"
    #更新记忆
    elif action == "update":
        if (
                not isinstance(index, int)
                or index < 0
                or index >= len(memory)
        ):
            return "Memory index 无效"
        memory[index] = new_memory
        save_memory(memory)
        return "旧记忆更新成功"

    return "无法判断记忆类型"


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
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个整数的乘法",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "保存用户长期稳定且未来有用的信息。如果用户一次提供多条独立信息，应分别保存。",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": [
                            "profile",
                            "preference",
                            "learning",
                            "project",
                            "other"
                        ]
                    },
                    "content": {
                        "type": "string"
                    }
                },
                "required": [
                    "memory_type",
                    "content"
                ]
            }
        }
    }
]

tool_map = {
    "calculator": calculator,
    "multiply": multiply,
    "remember": remember,
}

messages = [
    {
        "role": "system",
        "content": build_system_prompt()
    }
]

MAX_HISTORY = 10

def trim_messages():
    global messages

    system_message = messages[0]

    history = []

    for msg in messages[1:]:
        if msg.get("role") == "user":
            history.append(msg)

        elif (
            msg.get("role") == "assistant"
            and msg.get("content")
            and not msg.get("tool_calls")
        ):
            history.append(msg)

    history = history[-MAX_HISTORY:]

    messages = [system_message] + history


def run_agent(user_input):
    relevant_memory = retrieve_memory(user_input)

    memory_text = json.dumps(
        relevant_memory,
        ensure_ascii=False,
        indent=2
    )

    messages[0]["content"] = f"""
    你是一个计算与学习 Agent。
    你可以调用工具完成任务。
    只有当用户提供长期稳定、未来仍然有帮助的信息时，
    才调用 remember 工具。
    如果用户一次提供多条独立的长期信息，
    可以分别调用 remember 工具保存。
    不要保存临时事件和一次性信息。
    与当前问题相关的长期 Memory：
    {memory_text}
    回答时不要使用 Markdown 加粗。
    """
    messages.append({
        "role": "user",
        "content": user_input
    })

    max_steps = 10

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools
        )

        message = response.choices[0].message

        if not message.tool_calls:
            final_answer = message.content

            messages.append({
                "role": "assistant",
                "content": final_answer
            })

            trim_messages()

            return final_answer

        messages.append(
            message.model_dump(
                exclude_none=True
            )
        )

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name

            arguments = json.loads(
                tool_call.function.arguments
            )

            print(f"调用工具：{tool_name}")
            print(f"参数：{arguments}")

            tool_function = tool_map[tool_name]

            result = tool_function(
                **arguments
            )

            print(f"工具结果：{result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return "Agent 执行次数超过限制"


while True:
    question = input("你：")

    if question.strip().lower() in [
        "退出",
        "结束",
        "exit",
        "quit"
    ]:
        print("AI：再见")
        break

    answer = run_agent(question)

    print("AI：", answer)
