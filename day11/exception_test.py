#异常处理试验

#模拟一个任务字典
task_data = {
    "task":"完成python作业",
    "priority":"high"
}

try:
    #尝试读取不存在的字段
    deadline = task_data["deadline"]

    print("截止时间：",deadline)

except KeyError as error:
    print("发生KeyError")
    print("缺少字段：",error)

print("程序继续运行:")