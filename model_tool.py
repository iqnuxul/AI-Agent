import torch
import torch.nn as nn
import pandas as pd
import joblib
import json

# 定义模型结构
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# 预测函数
def predict_sales(input_dict: dict) -> float:
    # 加载列顺序
    with open('input_columns.json', 'r') as f:
        input_columns = json.load(f)

    # 构造 DataFrame
    df = pd.DataFrame([input_dict])

    # One-hot 编码
    df = pd.get_dummies(df, columns=['StateHoliday'], drop_first=True)

    # 保证列顺序一致
    for col in input_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[input_columns]

    # 标准化
    scaler = joblib.load('scaler.pkl')
    input_scaled = scaler.transform(df)
    input_tensor = torch.FloatTensor(input_scaled)

    # 加载模型
    model = MLP(input_tensor.shape[1])
    model.load_state_dict(torch.load('mlp_model.pth'))
    model.eval()

    # 预测
    with torch.no_grad():
        prediction = model(input_tensor).item()

    return prediction
