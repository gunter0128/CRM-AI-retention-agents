# src/data_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def prepare_data():
    # 1. 讀原始資料
    raw_path = Path("data/raw/Telco-Customer-Churn.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"找不到 {raw_path}，請確認檔案有放對位置")

    df = pd.read_csv(raw_path)

    # 2. 建立二元標籤欄位：ChurnLabel (Yes -> 1, No -> 0)
    df["ChurnLabel"] = (df["Churn"] == "Yes").astype(int)

    # 3. 選一些我們要拿來當特徵的欄位
    feature_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
    ]
    target_col = "ChurnLabel"

    # 有些 TotalCharges 會是空字串，要先處理掉
    df = df.replace(" ", pd.NA)
    df = df.dropna(subset=["TotalCharges"])

    # 把 TotalCharges 轉成數值
    df["TotalCharges"] = df["TotalCharges"].astype(float)

    # 4. 做 One-Hot Encoding：把類別變數展開成 0/1 欄位
    X = pd.get_dummies(df[feature_cols])
    y = df[target_col]

    # 5. 存一份「訓練用的 features + label + customerID」
    processed = X.copy()
    processed["ChurnLabel"] = y.values
    processed["customerID"] = df["customerID"].values

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / "churn_features.csv"
    processed.to_csv(processed_path, index=False)
    print(f"已輸出訓練資料到 {processed_path}")

    # 6. 再存一份「比較原始的客戶 profile」給後面 LLM agents 看
    profile_cols = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "PaymentMethod",
    ]
    profiles_path = processed_dir / "customer_profiles.csv"
    df[profile_cols].to_csv(profiles_path, index=False)
    print(f"已輸出客戶資料到 {profiles_path}")


if __name__ == "__main__":
    prepare_data()
