import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.optim.lr_scheduler as lr_scheduler

# 1. 定義資料集
class StockDataset(Dataset):
    def __init__(self, csv_file):
        # 使用 Big5 編碼讀取 CSV（根據實際編碼調整）
        self.data = pd.read_csv(csv_file, encoding='big5')
        # 清理欄位名稱，移除前後空格
        self.data.columns = [col.strip() for col in self.data.columns]
        # 依日期排序（若有 date 欄位）
        if 'date' in self.data.columns:
            self.data = self.data.sort_values(by='date').reset_index(drop=True)
        print("CSV 欄位名稱：", self.data.columns.tolist())
        
        # 對 "amount" 欄位進行正規化：先移除逗號，再轉換為 float，
        # 並使用 feature_range=(0, 10)（這裡你可以根據需求調整範圍）
        self.data['amount'] = self.data['amount'].apply(lambda x: float(str(x).replace(',', '')))
        scaler_amount = MinMaxScaler(feature_range=(0, 10))
        self.data[['amount']] = scaler_amount.fit_transform(self.data[['amount']])
        # 保存 scaler_amount 方便後續逆轉換
        self.scaler_amount = scaler_amount
        
        # 對 "deal" 欄位也進行正規化（feature_range=(0, 1)）
        self.data['deal'] = self.data['deal'].apply(lambda x: float(str(x).replace(',', '')))
        scaler_deal = MinMaxScaler(feature_range=(0, 1))
        self.data[['deal']] = scaler_deal.fit_transform(self.data[['deal']])
        
    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 將輸入數據拆分為兩部分：
        # 分支1: 處理 amount（已正規化）
        input_cols_amount = ['amount']
        # 分支2: 其他特徵，這裡排除 money
        input_cols_other = ['open', 'max', 'min', 'close', 'delta', 'deal']
        
        amount_val = self.data.iloc[idx][input_cols_amount].values.astype('float32')
        other_vals = self.data.iloc[idx][input_cols_other].apply(
            lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x
        ).values.astype('float32')
        
        features = {
            'amount': torch.tensor(amount_val),
            'other': torch.tensor(other_vals)
        }
        # 目標：預測下一日的 [amount, open, max, min, close]
        target_cols = ['amount', 'open', 'max', 'min', 'close']
        target = self.data.iloc[idx + 1][target_cols].apply(
            lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x
        ).values.astype('float32')
        
        return features, torch.tensor(target)


# 2. 定義多分支神經網路模型
class MultiBranchStockPredictor(nn.Module):
    def __init__(self, other_input_size=6, hidden_size=32, output_size=5):
        super(MultiBranchStockPredictor, self).__init__()
        # 分支1：處理 amount (1 個特徵)
        self.amount_branch = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU()
        )
        # 分支2：處理其他特徵 (6 個特徵)
        self.other_branch = nn.Sequential(
            nn.Linear(other_input_size, 8),
            nn.ReLU()
        )
        # 整合兩分支後接全連接層，輸出 5 維 (amount, open, max, min, close)
        self.combined_fc = nn.Sequential(
            nn.Linear(16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, features):
        amount_input = features['amount']   # shape: (batch, 1)
        other_input = features['other']       # shape: (batch, 6)
        
        out_amount = self.amount_branch(amount_input)
        out_other = self.other_branch(other_input)
        
        combined = torch.cat((out_amount, out_other), dim=1)
        output = self.combined_fc(combined)
        return output


if __name__ == '__main__':
    csv_file = r'D:\TradePredictor\data\STOCK_DAY_2002.csv'
    dataset = StockDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiBranchStockPredictor(other_input_size=6, hidden_size=64, output_size=5)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 使用學習率調度器，每 10 個 epoch 將學習率衰減至原來的 0.95 倍
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    num_epochs = 1000
    threshold = 0.72
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
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        scheduler.step()
        if avg_loss < threshold:
            print(f"Loss has approached {threshold}, stopping training early.")
            break

    # 使用最新一天的資料預測明日目標值
    last_row = dataset.data.iloc[-1]
    input_cols_amount = ['amount']
    input_cols_other = ['open', 'max', 'min', 'close', 'delta', 'deal']
    amount_val = last_row[input_cols_amount].values.astype('float32')
    other_vals = last_row[input_cols_other].apply(
        lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x
    ).values.astype('float32')

    features = {
        'amount': torch.tensor(amount_val).unsqueeze(0),  # 增加 batch 維度
        'other': torch.tensor(other_vals).unsqueeze(0)
    }

    model.eval()
    with torch.no_grad():
        prediction = model(features)

    pred_np = prediction.numpy().flatten()
    target_cols = ['預測明日amount', '預測明日open', '預測明日max', '預測明日min', '預測明日close']
    
    # 將預測的 normalized amount（pred_np[0]）轉換回原始尺度
    normalized_amount = pred_np[0]
    original_amount = dataset.scaler_amount.inverse_transform([[normalized_amount]])[0][0]
    print(f"預測明日amount(原始尺度）：{original_amount:.2f}")
    
    # 其餘預測值直接輸出
    for col, val in zip(target_cols[1:], pred_np[1:]):
        print(f"{col}: {val:.2f}")
