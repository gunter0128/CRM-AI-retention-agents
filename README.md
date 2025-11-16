# CRM AI Agents

### 流失預測 × 原因推論 × 客戶挽留策略 × AI 多代理助理

> 使用「自行訓練的流失預測模型（Churn Model）＋CRM 客戶資料分析＋LLM 多代理（Agents）」
> 自動化產生 **客戶流失風險分析、推論原因、挽留方案與溝通內容**，支援 CRM/客服/行銷部門日常決策。

---

# 1. 為什麼做這個主題？

因為我要應徵的是 **Business AI Engineer（TSMC BSID）**，
主管們強調：

* **不能只會叫 API，要能夠自己做資料、建模型、整流程**
* **要懂企業流程（SCM / CRM）如何與 AI 整合**
* **要把模型輸出的東西變成「可用的決策工具」**
* **要能做成可以 Demo 的成品**

這個專案可以完整展示：

* 對 **CRM 客戶資料（Telco Churn）** 的 ETL 與前處理能力
* 自行訓練的 baseline 流失預測模型（Logistic Regression，可替換 XGBoost）
* 多個 LLM Agents 分工協作
* 一條從資料 → 模型 → 推論 → 決策 → 文案 → UI 的完整 AI 流程
* Streamlit Dashboard 讓主管可以直接操作 Demo

專案目的：
**讓面試官清楚看到我能將資料處理、ML、業務規則、LLM、多代理、前端整合成一套真正能用的企業 AI 系統。**

---

# 2. 系統架構概觀

整體由四層組成：

---

## 2.1 資料處理層（Data Prep）

來源資料：
**Kaggle – Telco Customer Churn dataset**

工作：

* 缺失值處理
* 類別編碼（One-hot / Label Encoding）
* 數值欄位清洗（TotalCharges）
* 產生處理後資料：`churn_features.csv`

---

## 2.2 流失預測層（Churn Model）

模型：

* Logistic Regression（baseline，可替換成 XGBoost / RandomForest）

輸出：

* 流失機率（0~1）
* 特徵影響指標（feature importance）
* 分數可用於風險切分（高、中、低）

### Risk Level（本專案設計）

| 機率區間    | 等級  |
| ------- | --- |
| ≥ 0.7   | 高風險 |
| 0.4–0.7 | 中風險 |
| < 0.4   | 低風險 |

---

## 2.3 客戶價值分群（Customer Value Segmentation）

本專案使用簡單但企業常用的「月租金額」區分：

| 月租金額 MonthlyCharges | 客戶價值 |
| ------------------- | ---- |
| ≥ 80                | 高價值  |
| 40–79               | 中價值  |
| < 40                | 低價值  |

（面試時可講：實務上會替換成 CLV / 盈利 / 行為價值模型）

---

## 2.4 AI Agents 層（LLM Multi-Agents）

專案共分為 **四個 AI Agents**：

### **Agent 1 — Data Analyst Agent**

* 使用 churn model 預測流失機率
* 整理「流失風險評估」與「特徵影響」

### **Agent 2 — Churn Reasoning Agent**

* 將分析結果轉成可讀、可解釋的原因說明
* 整理出「主要流失原因總結」

### **Agent 3 — Campaign Designer Agent**

* 依照客戶價值（高/中/低）
* 依照流失原因
* 自動產生挽留方案（含成本＆適合原因）

### **Agent 4 — Communication Agent**

* 根據方案產生：

  * Email 文案（完整一封）
  * SMS 簡訊（70 字）
  * 電話話術（客服可照著念）

---

# 3. 使用資料集：Telco Customer Churn（Kaggle）

來源：
[https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)

專案主要使用：

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

請手動下載並放入：

```
data/raw/
```

---

# 4. 專案目錄結構

```
crm-ai-agents/
├── data/
│   ├── raw/                        # 原始客戶資料（需自行下載）
│   └── processed/                  # 前處理後資料（churn_features.csv）
│
├── models/
│   └── churn_model.pkl             # 訓練後的流失預測模型
│
├── src/
│   ├── data_prep.py                # 資料清洗 + 特徵工程
│   ├── train_churn_model.py        # 訓練 Logistic Regression churn model
│   ├── tools.py                    # 資料查詢、模型推論、特徵取得工具
│   ├── pipeline.py                 # 4 個 Agents 串成一條流程
│   │
│   ├── agents/
│   │   ├── data_analyst.py         # Agent 1：流失風險分析
│   │   ├── churn_reasoning.py      # Agent 2：原因推論
│   │   ├── campaign_designer.py    # Agent 3：挽留策略
│   │   └── communication.py        # Agent 4：Email / SMS / 話術生成
│   │
│   └── dashboard.py            # Streamlit Dashboard（demo 用）
│
├── .gitignore
└── README.md
```

---

# 5. 如何重現專案

## 5.1 建立環境

```bash
git clone https://github.com/<your-account>/CRM-AI-retention-agents.git
cd CRM-AI-retention-agents

python -m venv venv
venv/Scripts/activate      # Windows
# source venv/bin/activate # Mac/Linux

pip install -r requirements.txt
```

---

## 5.2 放置資料集

請將 Churn Dataset 放到：

```
data/raw/
```

---

## 5.3 執行資料前處理

```bash
python -m src.data_prep
```

---

## 5.4 訓練 Churn Model

```bash
python -m src.train_churn_model
```

會產生：

```
models/churn_model.pkl
```

---

## 5.5 測試端到端 Pipeline（指令列 Demo）

```bash
python test_pipeline.py
```

---

## 5.6 啟動 Streamlit Dashboard（重點 Demo）

```bash
streamlit run src/app_streamlit.py
```

功能：

* 客戶基本資料
* 流失機率（含高/中/低風險）
* Agent 1：流失風險評估摘要 + 詳細內容
* Agent 2：主要流失原因摘要 + 詳細內容
* Agent 3：挽留策略摘要 + 詳細方案
* Agent 4：Email / SMS / 電話話術

---

# 6. 模型與規則的可解釋性設計

## 6.1 Churn Model（Logistic Regression）

* 簡單、可解釋
* 特徵影響方向明確
* 可快速迭代 / 替換成更強模型（XGBoost）

## 6.2 風險等級切分

* 高風險：p ≥ 0.7
* 中風險：0.4 ≤ p < 0.7
* 低風險：p < 0.4

可依企業 SLA / retention policy 調整。

## 6.3 客群價值分群

基於 `MonthlyCharges`：

| 月租金額  | 等級  |
| ----- | --- |
| ≥ 80  | 高價值 |
| 40–79 | 中價值 |
| < 40  | 低價值 |

（可替換成 CLV、使用量、營收等方法）

---

# 7. 企業價值（Business Impact）

### **① 自動化流失預測流程**

CRM 團隊不用再人工查表、算數據。

### **② 可讀、可解釋的原因分析**

業務或客服可以直接使用。

### **③ 自動產生挽留方案**

依價值與原因動態調整策略。

### **④ 自動產生客戶溝通內容**

直接可用於：

* 電話客服
* 行銷信
* 簡訊回訪

### **⑤ 整合到企業數據流程容易**

Pipeline 與 Agents 都模組化：

* 可放進 CRM 系統 API
* 可由 batch job 每天自動跑
* 可對接行銷自動化系統（MA）

