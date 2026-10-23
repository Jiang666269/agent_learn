#整合最小工作流

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

#定义工作流状态
class TaskState(TypedDict):
    task: str
    priority: str
    result: str

#根据规则判断优先级
def classify_task(state: TaskState):
    task = state["task"]

    if "故障" in task or "紧急" in task:
        priority = "high"
    else:
        priority = "normal"

    print(f"任务分类完成：{priority}")

    return {"priority": priority}

#紧急任务处理节点
def urgent_handler(state: TaskState):
    print("进入紧急处理节点")

    return {
        "result": f"需要优先处理：{state['task']}"
    }

#普通任务处理节点
def normal_handler(state: TaskState):
    print("进入普通处理节点")

    return {
        "result": f"加入待办列表：{state['task']}"
    }

#条件路由函数
def route_task(state: TaskState):
    if state["priority"] == "high":
        return "urgent"

    return "normal"

#创建 StateGraph
builder = StateGraph(TaskState)

#注册节点
builder.add_node("classify", classify_task)
builder.add_node("urgent_handler", urgent_handler)
builder.add_node("normal_handler", normal_handler)

#固定入口
builder.add_edge(START, "classify")

#根据 State 动态路由
builder.add_conditional_edges(
    "classify",
    route_task,
    {
        "urgent": "urgent_handler",
        "normal": "normal_handler"
    }
)

#定义结束路径
builder.add_edge("urgent_handler", END)
builder.add_edge("normal_handler", END)

#编译
graph = builder.compile()

#用户输入
user_task = input("请输入任务：").strip()

#执行
result = graph.invoke({
    "task": user_task,
    "priority": "",
    "result": ""
})

print("\n最终 State：")
print(result)

