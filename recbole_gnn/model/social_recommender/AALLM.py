# -*- coding:utf-8 -*-

import openai

openai.api_key = '9b09f2e4c1d62af215d0b85d9bc6def4'
openai.api_base = 'https://gateway.chat.sensedeal.vip/v1'

# q = "你是一个linux大师。现在home目录空间不足，要求对其进行扩容，/data目录空间足够，如何将/data目录的部分空间扩充到home目录下？请给出具体操作流程。使用的是 MBR"
q = "hello"
with open("AAquestion.txt", "r", encoding='utf-8') as f:  # 打开文件
    q = f.read()  # 读取文件

rsp = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": q},
        {"role": "user", "content": q}
    ]
)

print(rsp.get("choices")[0]["message"]["content"])