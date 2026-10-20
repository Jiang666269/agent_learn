#依赖 jsonschema

from jsonschema import validate,ValidationError


#定义Schema
task_schema = {
    "type": "object",
    "properties": {
        "task": {"type": "string"},
        "priority": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "deadline": {"type": "string"}
    },
    "required": ["task", "priority", "deadline"],
    "additionalProperties": False #是否允许额外返回
}

#模拟模型返回的数据
task_data = {
    "task":"完成python作业",
    "priority":"high",
    "deadline":"明天下午五点"
}

#校验数据
try:
    validate(instance=task_data,schema=task_schema) #按照schema检查数据是否符合要求
    print("验证成功，数据符合JSON Schema")

except ValidationError as error:
    print("验证失败：")
    print(error.message)  #获取简洁的校验错误信息