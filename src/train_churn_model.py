# src/train_churn_model.py

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split


def train_model():
    processed_path = Path("data/processed/churn_features.csv")
    if not processed_path.exists():
        raise FileNotFoundError(
            f"找不到 {processed_path}，請先執行 python -m src.data_prep"
        )

    df = pd.read_csv(processed_path)

    # y = label，X = 特徵（把 label 和 customerID 拿掉）
    y = df["ChurnLabel"]
    X = df.drop(columns=["ChurnLabel", "customerID"])

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # 讓 0/1 比例維持差不多
    )

    # 建一個簡單的 Logistic Regression 當 baseline churn model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 評估一下 AUC + 簡單分類報告
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)

    print("=== Churn Model Evaluation ===")
    print(f"Test AUC: {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # 儲存 model 與特徵欄位順序
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "churn_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n已將模型存到 {model_path}")

    feature_cols_path = models_dir / "feature_columns.json"
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False, indent=2)
    print(f"已將特徵欄位順序存到 {feature_cols_path}")


if __name__ == "__main__":
    train_model()
