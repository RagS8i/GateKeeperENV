"""Quick connectivity check: sends one message to your model and prints the reply.

    API_KEY=... BASE_URL=... MODEL=... python check_model.py
"""
import os

import openai

client = openai.OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"],
)

response = client.chat.completions.create(
    model=os.environ["MODEL"],
    messages=[{"role": "user", "content": "Hi, How are you?"}],
)

print(response)
