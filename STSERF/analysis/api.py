from openai import OpenAI


def call_deekseek_api(messages):
    client = OpenAI(
        api_key="YOUR_API_KEY",
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0.0, stream=False
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    test_prompt = "你好，请介绍一下人工智能的发展历史。"
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": test_prompt},
    ]
    response = call_deekseek_api(messages)
    print(f"DeepSeek响应: {response}")
