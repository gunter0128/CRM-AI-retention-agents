# src/agents/communication.py

from . import call_llm
from src.tools import query_customer_profile


SYSTEM_PROMPT = """
你是一位擅長對客戶溝通的行銷與客服專家。
你會根據「客戶資料」與「行銷挽留方案」，幫團隊產出：

- Email 內容（完整一封）
- 簡訊內容（70 字以內的簡短版）
- 客服電話話術（客服人員可以照著念）

請使用自然、禮貌、不浮誇的繁體中文，避免過度承諾。
"""


def generate_communications(customer_id: str, campaign_result: dict) -> dict:
    """
    輸入：
        - customer_id
        - campaign_result: Campaign Designer Agent 的輸出 dict
          預期包含：
            - "campaign_plan": str
            - "value_segment": str （高價值 / 中價值 / 低價值）

    輸出：
        {
            "customer_id": ...,
            "communications": "包含 Email / SMS / Call script 的文字"
        }
    """
    profile = query_customer_profile(customer_id)
    campaign_plan = campaign_result.get("campaign_plan", "")
    value_segment = campaign_result.get("value_segment", "未分群")

    user_prompt = f"""
你會收到一位電信客戶的基本資料與針對他的「挽留方案」說明，請你幫忙撰寫對外溝通內容。

【客戶編號】
{customer_id}

【客戶價值分群】
{value_segment}

【客戶資料（欄位=值）】
{profile}

【行銷挽留方案說明】
{campaign_plan}

請你用繁體中文，依照以下格式產出三種內容：

一、Email 內容
- 請幫我寫一封可以直接寄給客戶的 Email。
- 包含：稱呼（例如「親愛的客戶您好」或客戶稱謂）、說明我們觀察到的情況（不要說「你快要流失」之類負面字眼）、提出適合他的方案、引導他採取下一步行動（如：登入帳號、點擊連結、洽詢客服）。

二、簡訊內容（SMS）
- 請寫一則 70 字以內的簡短簡訊版本。
- 保持禮貌與清楚，重點說明有優惠或方案可以協助他。

三、客服電話話術
- 請寫一份電話開場與溝通稿，大約 5~8 句話。
- 讓客服人員可以自然地開場、說明來意、提出方案、詢問客戶意願。
- 注意不要讓客戶覺得被威脅或情緒勒索。

請清楚區分三個部分，並使用適合商業溝通的語氣。
"""

    text = call_llm(SYSTEM_PROMPT, user_prompt)

    return {
        "customer_id": customer_id,
        "communications": text,
    }
