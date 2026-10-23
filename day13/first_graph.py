#LangGraph
#python -m pip install -U langgraph
#StateGraph->add_node->add_edge->compile->invoke

from typing import TypedDict
from langgraph.graph import StateGraph,START,END

#定义工作流state
#typedict 用于类型标注
class AgentState(TypedDict):
    message:str
    result:str

#定义第一个节点
#接收state 返回更新内容
def process_message(state:AgentState):
    user_message =  state["message"]

    return {
        "result":f"已处理信息：{user_message}"
    }

#创建工作流
builder = StateGraph(AgentState)

#注册节点
builder.add_node("process",process_message)

#连接执行路径
#决定执行顺序
builder.add_edge(START,"process")
builder.add_edge("process",END)

#编译工作流
graph = builder.compile()

#运行
result = graph.invoke({
    "message":"学习LangGraph",
    "result":""
})

print(result)