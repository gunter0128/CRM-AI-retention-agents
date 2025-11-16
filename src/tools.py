# src/tools.py

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd


DATA_PROCESSED_PATH = Path("data/processed/churn_features.csv")
PROFILE_PATH = Path("data/processed/customer_profiles.csv")
MODEL_PATH = Path("models/churn_model.pkl")
FEATURE_COLS_PATH = Path("models/feature_columns.json")


@lru_cache(maxsize=1)
def _load_churn_df() -> pd.DataFrame:
    """載入含有特徵 + 標籤 + customerID 的資料表"""
    if not DATA_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {DATA_PROCESSED_PATH}，請先執行 python -m src.data_prep"
        )
    return pd.read_csv(DATA_PROCESSED_PATH)


@lru_cache(maxsize=1)
def _load_profiles_df() -> pd.DataFrame:
    """載入比較原始的客戶 profile（給 LLM 看的）"""
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {PROFILE_PATH}，請先執行 python -m src.data_prep"
        )
    return pd.read_csv(PROFILE_PATH)


@lru_cache(maxsize=1)
def _load_churn_model():
    """載入訓練好的 churn model"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {MODEL_PATH}，請先執行 python -m src.train_churn_model"
        )
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_feature_cols() -> List[str]:
    """載入當初訓練時的特徵欄位順序"""
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {FEATURE_COLS_PATH}，請先執行 python -m src.train_churn_model"
        )
    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
        cols = json.load(f)
    return cols


def list_customer_ids() -> List[str]:
    """回傳所有 customerID 清單（給之後 UI 下拉選單用）"""
    df = _load_churn_df()
    return df["customerID"].tolist()


def query_customer_profile(customer_id: str) -> Dict:
    """回傳某個客戶的 profile（原始欄位為主）"""
    profiles = _load_profiles_df()
    row = profiles[profiles["customerID"] == customer_id]
    if row.empty:
        raise ValueError(f"找不到 customerID={customer_id} 的客戶")
    return row.iloc[0].to_dict()


def predict_churn(customer_id: str) -> float:
    """
    使用訓練好的模型，對指定 customerID 預測流失機率（回傳 0~1 間的浮點數）
    """
    df = _load_churn_df()
    feature_cols = _load_feature_cols()
    model = _load_churn_model()

    row = df[df["customerID"] == customer_id]
    if row.empty:
        raise ValueError(f"找不到 customerID={customer_id} 的客戶")

    X = row[feature_cols]
    prob = float(model.predict_proba(X)[0, 1])
    return prob


def get_random_customer_id() -> str:
    """從資料集中隨機挑一位客戶（之後 demo 可以用）"""
    df = _load_churn_df()
    return df.sample(1)["customerID"].iloc[0]
