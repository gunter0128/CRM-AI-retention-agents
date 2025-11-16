# src/dashboard
# .py

import sys
import os
import re

# 把專案根目錄加入 Python 路徑，讓 `import src.xxx` 可以正常運作
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd

from src.tools import (
    list_customer_ids,
    query_customer_profile,
    get_random_customer_id,
)
from src.pipeline import run_full_pipeline


def risk_level(prob: float) -> str:
    """根據機率給一個簡單的風險等級標籤"""
    if prob >= 0.7:
        return "高風險"
    elif prob >= 0.4:
        return "中風險"
    else:
        return "低風險"


def extract_numbered_section(text: str, section_no: int = 1) -> str:
    """
    從 LLM 的輸出中抽出「第 N 點」的內容：
    - 假設內容是這種格式：
        1. xxx
           xxx
        2. yyy
    - 我們會擷取從 `N.` 開頭那一行開始
      一直到下一個 `數字.` 開頭的行之前。
    """
    if not text:
        return ""

    lines = text.splitlines()
    start_prefix = f"{section_no}."
    captured = []
    capturing = False

    for line in lines:
        stripped = line.strip()
        if not capturing:
            if stripped.startswith(start_prefix):
                capturing = True
                captured.append(stripped)
        else:
            # 如果遇到新的「數字.」開頭，就代表下一段開始了
            if re.match(r"^[0-9]+\.", stripped):
                break
            captured.append(stripped)

    if not captured:
        # 如果沒抓到，就退而求其次回傳全文
        return text.strip()

    return "\n".join(captured).strip()


def main():
    st.set_page_config(
        page_title="AI CRM Retention Agents Demo",
        layout="wide",
    )

    # 自訂按鈕樣式
    st.markdown(
        """
        <style>
        .stButton > button {
            background-color: #ff7f50;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.4rem 0.8rem;
        }
        .stButton > button:hover {
            background-color: #ff9966;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI CRM Retention Agents Demo")
    st.caption("以多個 AI agents 協助 CRM 團隊分析客戶流失風險、推論原因，並設計挽留方案與溝通內容。")

    # --- Sidebar: 客戶選擇 ---

    st.sidebar.header("客戶選擇")

    try:
        customer_ids = list_customer_ids()
    except Exception as e:
        st.sidebar.error(f"載入客戶清單時發生錯誤：{e}")
        return

    if not customer_ids:
        st.sidebar.error("沒有可用的客戶資料。請確認前處理與模型訓練是否已完成。")
        return

    # 初始化目前選擇的客戶 ID
    if "selected_customer_id" not in st.session_state:
        st.session_state["selected_customer_id"] = customer_ids[0]

    # 先處理「隨機挑一位客戶」按鈕：按下時直接更新 session_state
    if st.sidebar.button("隨機挑一位客戶"):
        random_id = get_random_customer_id()
        st.session_state["selected_customer_id"] = random_id
        st.sidebar.success(f"已隨機選擇客戶：{random_id}")

    # selectbox 使用目前 session_state 裡的 selected_customer_id 做預設
    current_id = st.session_state["selected_customer_id"]
    if current_id not in customer_ids:
        current_id = customer_ids[0]

    selected_id = st.sidebar.selectbox(
        "選擇一位客戶",
        options=customer_ids,
        index=customer_ids.index(current_id),
    )
    # 使用者手動選擇時，更新 session_state
    st.session_state["selected_customer_id"] = selected_id

    st.sidebar.markdown("---")
    st.sidebar.write("點擊下方按鈕執行完整 AI agents pipeline：")

    run_button = st.sidebar.button("開始分析這位客戶")

    # --- 主畫面內容 ---
    if not run_button:
        st.info("請在左側選擇客戶，並按下「開始分析這位客戶」。")
        return

    # 真正用來分析的 ID（一定是最新的）
    selected_id = st.session_state["selected_customer_id"]

    # 執行 pipeline
    with st.spinner("AI agents 正在分析中，請稍候..."):
        try:
            result = run_full_pipeline(selected_id)
        except Exception as e:
            st.error(f"執行 pipeline 時發生錯誤：{e}")
            return

    # 1. 客戶總覽與流失風險
    st.subheader("1️⃣ 客戶總覽與流失風險")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**客戶基本資料**")
        profile = query_customer_profile(selected_id)
        df_profile = pd.DataFrame([profile]).T
        df_profile.columns = ["值"]
        st.table(df_profile)

    with col2:
        prob = result["analyst"]["churn_probability"]
        level = risk_level(prob)

        st.metric(
            label="預測流失機率",
            value=f"{prob:.3f}",
            delta=level,
            help="由自訓練 Logistic Regression churn model 預測出的流失機率（0~1）。",
        )

        with st.expander("風險等級說明（本 Demo 設計）"):
            st.markdown(
                """
                在這個 Demo 中，我們將 churn model 預測的流失機率粗略切成三個等級：

                - **高風險**：預測流失機率 ≥ 0.7  
                - **中風險**：0.4 ≤ 預測流失機率 < 0.7  
                - **低風險**：預測流失機率 < 0.4  

                實務上可以依照公司的容忍度與資源分配，重新調整這些 threshold，
                例如對高價值客戶設定更嚴格的門檻，或是依照不同方案做 A/B test 調整。
                """
            )

        # 這裡的簡介 = Data Analyst Output 中的「第 1 點（流失風險評估）」＋內容
        full_analyst = result["analyst"]["analysis"]
        analyst_section_1 = extract_numbered_section(full_analyst, section_no=1)

        st.markdown("**Data Analyst Agent：流失風險評估**")
        st.write(analyst_section_1)
        with st.expander("查看完整分析內容（包含指標與建議）"):
            st.write(full_analyst)

    st.markdown("---")

    # 2. 流失原因推論
    st.subheader("2️⃣ Churn Reasoning Agent：流失原因推論")

    full_reasoning = result["reasoning"]["reasoning"]
    # 這裡的簡介 = Churn Reasoning Output 中的「第 1 點（主要流失原因總結）」＋內容
    reasoning_section_1 = extract_numbered_section(full_reasoning, section_no=1)

    st.markdown("**主要流失原因總結**")
    st.write(reasoning_section_1)
    with st.expander("查看詳細流失原因說明（包含關鍵因素與影響）"):
        st.write(full_reasoning)

    st.markdown("---")

    # 3. 行銷挽留方案設計
    st.subheader("3️⃣ Campaign Designer Agent：挽留方案設計")

    col3, col4 = st.columns([1, 2])

    with col3:
        st.markdown("**客戶價值分群**")
        value_segment = result["campaign"]["value_segment"]
        st.write(value_segment)

        with st.expander("客戶價值分群說明（本 Demo 設計）"):
            st.markdown(
                """
                在這個 Demo 中，我們先用月租金額 `MonthlyCharges` 粗略區分客戶價值等級：

                - **高價值客戶**：MonthlyCharges ≥ 80  
                - **中價值客戶**：40 ≤ MonthlyCharges < 80  
                - **低價值客戶**：MonthlyCharges < 40  

                實務上可以進一步改成使用「客戶終身價值（CLV）」、「過去 12 個月營收」、
                「交叉銷售潛力」等更完整的指標來做分群。
                """
            )

    with col4:
        full_campaign = result["campaign"]["campaign_plan"]
        # 這裡的簡介 = 挽留方案說明中的「前言 / 概述」，也是假設 LLM 的第 1 點是思考邏輯
        campaign_section_1 = extract_numbered_section(full_campaign, section_no=1)

        st.markdown("**挽留方案設計概述**")
        st.write(campaign_section_1)
        with st.expander("查看完整挽留方案內容（各方案細節與成本考量）"):
            st.write(full_campaign)

    st.markdown("---")

    # 4. 對客戶的溝通內容
    st.subheader("4️⃣ Communication Agent：對客戶的溝通內容")

    full_comm = result["communications"]["communications"]
    st.markdown("**溝通內容摘要**")
    st.write("以下是完整 Email / 簡訊 / 電話話術內容。")
    with st.expander("查看完整 Email / 簡訊 / 電話話術內容"):
        st.write(full_comm)


if __name__ == "__main__":
    main()
