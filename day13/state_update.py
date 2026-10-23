#状态更新

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#定义共享的 State 结构
class AgentState(TypedDict):
    message: str
    task: str
    result: str

#第一个节点：提取任务
def extract_task(state: AgentState):
    print("进入 extract_task")
    print("收到的 message：", state["message"])

    return {
        "task": state["message"].strip()
    }

#第二个节点：处理任务
def process_task(state: AgentState):
    print("\n进入 process_task")
    print("读取到的 task：", state["task"])

    return {
        "result": f"任务已登记：{state['task']}"
    }

#创建图
builder = StateGraph(AgentState)

builder.add_node("extract_task", extract_task)
builder.add_node("process_task", process_task)

#定义执行路径
builder.add_edge(START, "extract_task")
builder.add_edge("extract_task", "process_task")
builder.add_edge("process_task", END)

#编译
graph = builder.compile()

#调用
result = graph.invoke({
    "message": "  学习 LangGraph 状态管理  ",
    "task": "",
    "result": ""
})

print("\n最终 State：")
print(result)
