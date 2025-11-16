# src/pipeline.py

from typing import Dict

from src.agents.data_analyst import analyze_customer
from src.agents.churn_reasoning import explain_churn_reason
from src.agents.campaign_designer import design_campaign
from src.agents.communication import generate_communications


def run_full_pipeline(customer_id: str) -> Dict:
    """
    給一個 customer_id，依序呼叫四個 Agent：

    1. Data Analyst Agent        -> 流失機率 + 指標分析
    2. Churn Reasoning Agent     -> 流失原因說明（故事化）
    3. Campaign Designer Agent   -> 挽留方案設計
    4. Communication Agent       -> Email / 簡訊 / 電話話術

    回傳一個 dict，結構大致如下：

    {
        "customer_id": "...",
        "analyst": { ... },
        "reasoning": { ... },
        "campaign": { ... },
        "communications": { ... }
    }
    """
    analyst = analyze_customer(customer_id)
    reasoning = explain_churn_reason(customer_id, analyst)
    campaign = design_campaign(customer_id, reasoning)
    communications = generate_communications(customer_id, campaign)

    return {
        "customer_id": customer_id,
        "analyst": analyst,
        "reasoning": reasoning,
        "campaign": campaign,
        "communications": communications,
    }
