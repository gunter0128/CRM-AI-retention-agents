# src/agents/__init__.py

import os
from typing import Literal, List, Dict

from openai import OpenAI

# 讀取環境變數中的 API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

Role = Literal["system", "user", "assistant"]


def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4.1-mini",  # 現在有的模型
) -> str:
    """
    統一封裝 LLM 呼叫邏輯，之後每個 agent 都呼叫這個函式。
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return resp.choices[0].message.content
