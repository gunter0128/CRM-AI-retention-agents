# src/agents/campaign_designer.py

from . import call_llm
from src.tools import query_customer_profile


SYSTEM_PROMPT = """
你是一位電信產業的行銷企劃，專門為「可能即將流失的客戶」設計挽留方案。

你很清楚：
- 折扣太多會傷害獲利
- 但完全不給優惠會失去高價值客戶
- 不同價值等級的客戶，投放資源的程度應該不同

請使用專業、務實、條理清楚的繁體中文回答。
"""


def estimate_customer_value(profile: dict) -> str:
    """
    根據客戶的月租金額粗略分群：
    - 高價值：MonthlyCharges >= 80
    - 中價值：40 <= MonthlyCharges < 80
    - 低價值：MonthlyCharges < 40
    這只是 demo 分法，面試時你可以說未來會改成更精細的 CLV 模型。
    """
    try:
        monthly = float(profile.get("MonthlyCharges", 0))
    except ValueError:
        monthly = 0.0

    if monthly >= 80:
        return "高價值"
    elif monthly >= 40:
        return "中價值"
    else:
        return "低價值"


def design_campaign(customer_id: str, churn_reasoning_result: dict) -> dict:
    """
    輸入：
        - customer_id
        - churn_reasoning_result: Churn Reasoning Agent 的輸出 dict
          預期至少包含 "reasoning": str

    輸出：
        {
            "customer_id": ...,
            "value_segment": "高價值 / 中價值 / 低價值",
            "campaign_plan": "自然語言描述的挽留方案"
        }
    """
    profile = query_customer_profile(customer_id)
    value_segment = estimate_customer_value(profile)
    reasoning_text = churn_reasoning_result.get("reasoning", "")

    user_prompt = f"""
你會收到一位電信客戶的資料，以及一份「流失原因說明」。
請你基於這些資訊，為該客戶設計合適的挽留方案。

【客戶編號】
{customer_id}

【客戶價值分群】
{value_segment} 客戶（依照月租費粗略判定）

【客戶資料（欄位=值）】
{profile}

【流失原因說明】
{reasoning_text}

請依照以下格式，設計 1~2 個主要挽留方案（用繁體中文回答）：

1. 先用 1~2 句話說明：你設計挽留方案時的思考邏輯（例如：高價值客戶可以給較有吸引力的方案，但仍要控管成本）。
2. 條列 1~2 個「挽留方案」，每個方案請包含：
   - 方案名稱
   - 方案內容（例如：幾折、幾個月、是否有升級或贈送服務）
   - 為什麼這個方案適合這位客戶（要跟上面的流失原因有關）
   - 成本與風險考量（簡短即可）
3. 最後給業務或行銷同仁一段 2~3 句話的建議，說明在執行這些方案時，需要注意什麼（例如：勿過度承諾、觀察後續使用行為變化等）。
"""

    campaign_text = call_llm(SYSTEM_PROMPT, user_prompt)

    return {
        "customer_id": customer_id,
        "value_segment": value_segment,
        "campaign_plan": campaign_text,
    }
