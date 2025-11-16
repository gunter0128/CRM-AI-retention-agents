# src/agents/data_analyst.py

from . import call_llm
from src.tools import predict_churn, query_customer_profile


SYSTEM_PROMPT = """
你是一位電信產業的資深數據分析師，專門分析客戶流失風險。
請用精簡、專業、讓業務與行銷同仁都看得懂的繁體中文做說明。
避免使用過度技術性的統計術語。
"""


def analyze_customer(customer_id: str) -> dict:
    """
    對指定 customer_id：
    1. 呼叫 churn model 預測流失機率
    2. 查詢客戶基本資料
    3. 用 LLM 產生一段「流失風險說明」

    回傳 dict，例如：
    {
        "customer_id": "...",
        "churn_probability": 0.83,
        "analysis": "文字說明..."
    }
    """
    prob = predict_churn(customer_id)
    profile = query_customer_profile(customer_id)

    # 給 LLM 的 user prompt
    user_prompt = f"""
以下是一位客戶的資料與流失預測結果：

- 客戶編號：{customer_id}
- 預測流失機率（0~1）：{prob:.3f}
- 客戶資料（欄位=值）：
{profile}

請你幫忙做一份「流失風險分析」，用繁體中文回答，內容包含：

1. 用一句話評估這位客戶的流失風險：高 / 中 / 低，並簡短說明理由。
2. 條列 3~5 個你認為較關鍵的指標與觀察（例如：tenure 長短、月租費高低、合約型態、TotalCharges 等），並解釋它們如何影響流失風險。
3. 給業務或客服一段 2~3 句話的建議，說明後續應該關注這位客戶的哪些行為或變化。
"""

    analysis_text = call_llm(SYSTEM_PROMPT, user_prompt)

    return {
        "customer_id": customer_id,
        "churn_probability": prob,
        "analysis": analysis_text,
    }
# src/agents/data_analyst.py

from . import call_llm
from src.tools import predict_churn, query_customer_profile


SYSTEM_PROMPT = """
你是一位電信產業的資深數據分析師，專門分析客戶流失風險。
請用精簡、專業、讓業務與行銷同仁都看得懂的繁體中文做說明。
避免使用過度技術性的統計術語。
"""


def analyze_customer(customer_id: str) -> dict:
    """
    對指定 customer_id：
    1. 呼叫 churn model 預測流失機率
    2. 查詢客戶基本資料
    3. 用 LLM 產生一段「流失風險說明」

    回傳 dict，例如：
    {
        "customer_id": "...",
        "churn_probability": 0.83,
        "analysis": "文字說明..."
    }
    """
    prob = predict_churn(customer_id)
    profile = query_customer_profile(customer_id)

    # 給 LLM 的 user prompt
    user_prompt = f"""
以下是一位客戶的資料與流失預測結果：

- 客戶編號：{customer_id}
- 預測流失機率（0~1）：{prob:.3f}
- 客戶資料（欄位=值）：
{profile}

請你幫忙做一份「流失風險分析」，用繁體中文回答，內容包含：

1. 用一句話評估這位客戶的流失風險：高 / 中 / 低，並簡短說明理由。
2. 條列 3~5 個你認為較關鍵的指標與觀察（例如：tenure 長短、月租費高低、合約型態、TotalCharges 等），並解釋它們如何影響流失風險。
3. 給業務或客服一段 2~3 句話的建議，說明後續應該關注這位客戶的哪些行為或變化。
"""

    analysis_text = call_llm(SYSTEM_PROMPT, user_prompt)

    return {
        "customer_id": customer_id,
        "churn_probability": prob,
        "analysis": analysis_text,
    }
