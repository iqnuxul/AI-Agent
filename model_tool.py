import torch
import torch.nn as nn
import pandas as pd
import joblib
import json

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def predict_sales(input_dict: dict) -> float:

    with open('input_columns.json', 'r') as f:
        input_columns = json.load(f)


    df = pd.DataFrame([input_dict])

    df = pd.get_dummies(df, columns=['StateHoliday'], drop_first=True)

    for col in input_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[input_columns]

    scaler = joblib.load('scaler.pkl')
    input_scaled = scaler.transform(df)
    input_tensor = torch.FloatTensor(input_scaled)

    model = MLP(input_tensor.shape[1])
    model.load_state_dict(torch.load('mlp_model.pth'))
    model.eval()

    with torch.no_grad():
        prediction = model(input_tensor).item()

    return prediction
