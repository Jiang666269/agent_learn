#重试机制
#具备 Max Attempts,Retry Condition,Retry Delay

import time

#模拟网络请求
def request_api(attempt):
    if attempt < 3:
        raise ConnectionError("模拟网络连接失败") #制造一个网络异常

    return "请求成功，已获得模型响应"

max_attempts = 3

for attempt in range(1, max_attempts+1):
    try:
        print(f"第{attempt}次尝试")

        result = request_api(attempt)
        print(result)
        break
    except ConnectionError as error:
        print(f"发生异常{error}")

        if attempt == max_attempts:
            print("已达最大尝试次数，quit")


        #指数退避
        delay = min(8,2 **(attempt-1))
        print(f"等待{delay}s后重试...")
        time.sleep(delay)
        
#SDK自身可能具备自动重试机制
#设置SDK的max_retries