# model_tool.py
import torch
import pandas as pd
import json
import joblib
from datetime import datetime
from typing import List, Dict, Union
from langchain.tools import tool

# 加载模型
model = torch.load("mlp_model.pth")
model.eval()

# 加载 scaler 和输入特征
scaler = joblib.load("scaler.pkl")
with open("input_columns.json", "r") as f:
    input_columns = json.load(f)

def preprocess_input(data: pd.DataFrame) -> torch.Tensor:
    """根据 input_columns 做特征补齐并标准化"""
    for col in input_columns:
        if col not in data.columns:
            data[col] = 0  # 缺失列补0
    data = data[input_columns]
    scaled = scaler.transform(data)
    return torch.tensor(scaled, dtype=torch.float32)

@tool
def predict_sales(store_data: List[Dict]) -> List[float]:
    """
    基于用户输入的店铺信息和日期预测销售额。
    输入示例:
    [
        {"Store": 1, "Date": "2015-06-01", "Promo": 1, "DayOfWeek": 1, "Open": 1},
        {"Store": 2, "Date": "2015-06-01", "Promo": 0, "DayOfWeek": 1, "Open": 1}
    ]
    返回预测销售额列表。
    """
    df = pd.DataFrame(store_data)
    
    # 处理日期字段
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    
    # 处理缺失值
    df.fillna(0, inplace=True)

    # 标准化 + 模型预测
    input_tensor = preprocess_input(df)
    with torch.no_grad():
        predictions = model(input_tensor).squeeze().tolist()
    return predictions


@tool
def recommend_promotions(store_list: List[int], date: str) -> Dict[int, str]:
    """
    为多个店铺在指定日期上推荐是否促销。
    返回 {store_id: "建议开启促销" 或 "无需促销"}。
    """
    recommendation = {}
    today = pd.to_datetime(date)
    weekday = today.dayofweek + 1  # 与数据一致，周一为 1

    # 构建模拟输入数据（分别带/不带促销）
    data = []
    for store in store_list:
        for promo in [0, 1]:
            data.append({
                "Store": store,
                "Date": date,
                "Promo": promo,
                "DayOfWeek": weekday,
                "Open": 1,
                "Customers": 500  # 默认估算，如你有更好方式可替代
            })

    df = pd.DataFrame(data)

    # 补充日期拆解
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    # 转换成模型输入
    input_tensor = preprocess_input(df)
    with torch.no_grad():
        pred = model(input_tensor).squeeze().tolist()

    # 每个店铺两个预测值：无促销、有促销
    for i, store in enumerate(store_list):
        sales_no_promo = pred[i * 2]
        sales_with_promo = pred[i * 2 + 1]
        if sales_with_promo - sales_no_promo > 300:  # 超过阈值就建议促销
            recommendation[store] = "建议开启促销"
        else:
            recommendation[store] = "无需促销"
    return recommendation
