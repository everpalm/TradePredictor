import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 1. 定義資料集，採用滑動視窗：以第 i 天作為輸入，第 i+1 天作為目標
class StockDataset(Dataset):
    def __init__(self, csv_file):
        # 使用 Big5 編碼讀取 CSV（根據你檔案實際編碼調整）
        self.data = pd.read_csv(csv_file, encoding='big5')
        # 清理欄位名稱，移除前後空格
        self.data.columns = [col.strip() for col in self.data.columns]
        # 依日期排序（若有 date 欄位）
        if 'date' in self.data.columns:
            self.data = self.data.sort_values(by='date').reset_index(drop=True)
        print("CSV 欄位名稱：", self.data.columns.tolist())

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 更新為 CSV 中的欄位名稱
        input_cols = ['amount', 'money', 'open', 'max', 'min', 'close',
                      'delta', 'deal']
        # 將每個欄位值轉換成 float，先移除逗號
        features = self.data.iloc[idx][input_cols].apply(
            lambda x: float(str(x).replace(',', ''))
        ).values.astype('float32')
        target_cols = ['amount', 'max', 'min', 'close']
        target = self.data.iloc[idx + 1][target_cols].apply(
            lambda x: float(str(x).replace(',', ''))
        ).values.astype('float32')

        return torch.tensor(features), torch.tensor(target)


# 2. 定義神經網路模型
class StockPredictor(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, output_size=4):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    # 讀取資料集
    csv_file = r'D:\TradePredictor\data\STOCK_DAY_2002_202503.csv'
    dataset = StockDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 建立模型、定義損失函數與優化器
    model = StockPredictor(input_size=8, hidden_size=32, output_size=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 訓練 (示範用 epoch 數較少，實際可依需要調整)
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for features, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{num_epochs},"
            "Loss: {epoch_loss/len(dataloader):.4f}"
        )

    # 以最新一天的資料預測明日目標值
    # 取 CSV 檔中最後一筆資料（注意：這筆資料尚無對應明日資料）
    # 取 CSV 檔中最後一筆資料（注意：這筆資料尚無對應明日資料）
    last_row = dataset.data.iloc[-1]
    input_cols = ['amount', 'money', 'open', 'max', 'min', 'close', 'delta',
                  'deal']
    # 先移除逗號再轉換成 float
    last_features = last_row[input_cols].apply(
        lambda x: float(str(x).replace(',', ''))
    ).values.astype('float32')
    last_features = torch.tensor(last_features).unsqueeze(0)  # 增加 batch 維度

    model.eval()
    with torch.no_grad():
        prediction = model(last_features)

    pred_np = prediction.numpy().flatten()
    target_cols = ['預測明日成交量', '預測明日max', '預測明日min', '預測明日close']
    for col, val in zip(target_cols, pred_np):
        print(f"{col}: {val:.2f}")
