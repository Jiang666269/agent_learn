#条件分支工作流

from typing import TypedDict
from langgraph.graph import StateGraph,START,END

#定义state
class AgentState(TypedDict):
    task:str
    priority:str
    result:str

#第一个节点，读取任务优先级
def check_priority(state:AgentState):
    print("正在检查任务优先级...")
    return {}

#紧急任务节点
def urgent_task(state:AgentState):
    return {
        "result":f"紧急处理：{state['task']}"
    }

#普通任务节点
def normal_task(state:AgentState):
    return {
        "result":f"加入代办：{state['task']}"
    }

#判断
def route_task(state:AgentState):
    if state["priority"] == "high":
        return "urgent"

    return "normal"

#创建工作流
builder = StateGraph(AgentState)

builder.add_node("check",check_priority)
builder.add_node("urgent_task",urgent_task)
builder.add_node("normal_task",normal_task)

#添加固定边
builder.add_edge(START,"check")

#条件边
builder.add_conditional_edges(
    "check",
    route_task,
    {
        "urgent":"urgent_task",
        "normal":"normal_task"
    }
)

#两条分支最后都到end
builder.add_edge("urgent_task",END)
builder.add_edge("normal_task",END)

#编译运行
graph = builder.compile()

result = graph.invoke({
    "task":"修复线上服务器故障",
    "priority":"high",
    "result":""
})

print(result)