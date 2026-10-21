#整合

import os
import json
import time
import random
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
    AuthenticationError,
)
from pydantic import BaseModel,ValidationError,ConfigDict

#定义数据模型
class Task(BaseModel):
    model_config = ConfigDict(extra="forbid",strict=True)

    task:str
    priority: Literal["high","medium","low"]
    deadline:str

#初始化Deepseek客户端
project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
    timeout = 10.0,
    max_retries=0
)

#网络请求，只对临时错误进行有限重试
def request_with_retry(messages,max_attempts=3):
    for attempt in range(1,max_attempts+1):
        try:
            return client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                response_format={"type":"json_object"}
            )

        except AuthenticationError:
            #API key 错误，重试无法解决
            raise

        except (APITimeoutError,APIConnectionError,RateLimitError) as error:
            last_error = error

        except APIStatusError as error:
            #对常见的临时服务端错误重试
            if error.status_code not in(500,502,503,504):
                raise
            last_error = error

        if attempt== max_attempts:
            raise last_error

        #指数退避+随机抖动
        delay = min(8.0,2 ** (attempt-1))
        delay += random.uniform(0,0.5)

        print(f"请求失败，{delay:.2f}s后重试...")
        time.sleep(delay)

#结构化任务提取+验证恢复
def extract_task(user_input,max_repairs=2):
    messages = [
        {
            "role":"system",
            "content":(
                "你是任务提取助手。"
                "必须输出合法 JSON 对象。"
                "JSON 必须包含 task、priority、deadline 三个字段。"
                "所有字段必须是字符串。"
                "priority 只能是 high、medium 或 low。"
                "deadline 不明确时使用空字符串。"
                "如果优先级缺乏判断依据，不要编造。"
            )
        },
        {
            "role":"user",
            "content":user_input
        }
    ]
    # 首次生成+最多两次格式恢复
    for repair_attempt in range(max_repairs + 1):
        response = request_with_retry(messages)

        content = response.choices[0].message.content

        try:
            data = json.loads(content)
            task = Task.model_validate(data)
            return task

        except (json.JSONDecodeError,ValidationError) as error:
            print(f"第{repair_attempt + 1}次输出验证失败")

            if repair_attempt == max_repairs:
                raise ValueError("模型输出多次不符合结构要求") from error

            #将错误信息反馈给模型
            messages.append({
                "role":"assistant",
                "content":content
            })

            messages.append({
                "role":"user",
                "content":(
                    f"上一次输出未通过验证：{error}\n"
                    "请根据原始任务重新生成合法 JSON。"
                    "只能包含 task、priority、deadline 三个字段。"
                    "priority 仅允许 high、medium、low。"
                    "不要添加其他说明。"
                )
            })

    raise RuntimeError("任务提取未能完成")

#main
if __name__ == "__main__":
    user_input = input("请输入任务：").strip()

    try:
        result = extract_task(user_input)

        print("\n提取成功:")
        print(result.model_dump_json(indent=2))

    except AuthenticationError:
        print("API认证失败，请检查DEEPSEEK_API_KEY")

    except (APITimeoutError,APIConnectionError,RateLimitError,APIStatusError) as error:
        print(f"API请求失败:{type(error).__name__}")

    except ValueError as error:
        print(f"数据验证失败：{error}")