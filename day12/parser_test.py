#StrOutputParser
#parser可以直接加入LCEL调用链

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

#模拟模型返回的信息
message = AIMessage(
    content = "pythobn装饰器用于在不修改原函数代码的情况下扩展功能"
)

#创建输出解析器
parser = StrOutputParser()

#解析模型信息
#将消息内容提取成字符串
result = parser.invoke(message)

#对比解析前后数据类型
print("解析前：",type(message))
print("解析后：",type(result))

print("\n解析结果:")
print(result)