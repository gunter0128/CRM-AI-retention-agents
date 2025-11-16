# src/agents/churn_reasoning.py

from . import call_llm
from src.tools import query_customer_profile


SYSTEM_PROMPT = """
你是一位熟悉電信產業的 CRM 顧問，擅長「用自然語言」解釋客戶可能流失的原因。
你會看數據分析師的說明與客戶資料，幫團隊整理出：
- 為什麼這位客戶可能會離開
- 哪些背景與行為模式值得注意
- 流失對公司的影響

請使用專業但好理解的繁體中文，條列清楚。
"""


def explain_churn_reason(customer_id: str, analyst_result: dict) -> dict:
    """
    輸入：
        - customer_id：客戶編號
        - analyst_result：Data Analyst Agent 的輸出 dict
          期待至少包含：
            - "churn_probability": float
            - "analysis": str

    輸出：
        {
            "customer_id": ...,
            "reasoning": "自然語言說明..."
        }
    """
    profile = query_customer_profile(customer_id)
    prob = analyst_result.get("churn_probability")
    analyst_text = analyst_result.get("analysis", "")

    user_prompt = f"""
你會收到一位客戶的資料與數據分析師的說明，請你幫忙進一步整理「流失原因」。

【客戶編號】
{customer_id}

【預測流失機率（0~1）】
{prob:.3f}

【客戶資料（欄位=值）】
{profile}

【數據分析師的說明】
{analyst_text}

請你產生一份「流失原因說明」，包含：

1. 用 2~3 句話總結這位客戶「可能會流失」的主要原因。
2. 條列 3~5 個關鍵因素，說明這些因素如何提高流失風險（例如：合約即將到期、帳單金額偏高、tenure 太短、使用率可能下降、付款方式帶來不確定性等）。
3. 簡短說明：如果這位客戶真的流失，對公司可能造成的影響（例如：高價值客戶流失、品牌評價、客服負擔等）。

請用繁體中文回答，條列清楚，不要寫太學術。
"""

    reasoning_text = call_llm(SYSTEM_PROMPT, user_prompt)

    return {
        "customer_id": customer_id,
        "reasoning": reasoning_text,
    }
