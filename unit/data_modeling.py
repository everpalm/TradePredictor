import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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
        
        # 對 "amount" 欄位進行正規化
        self.data['amount'] = self.data['amount'].apply(lambda x: float(str(x).replace(',', '')))
        scaler_amount = MinMaxScaler(feature_range=(0, 10))
        self.data[['amount']] = scaler_amount.fit_transform(self.data[['amount']])
        
        # 如果 "deal" 欄位需要正規化，也可一併處理（此處示範，如有需要可打開）
        self.data['deal'] = self.data['deal'].apply(lambda x: float(str(x).replace(',', '')))
        scaler_deal = MinMaxScaler(feature_range=(0, 1))
        self.data[['deal']] = scaler_deal.fit_transform(self.data[['deal']])
        
    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 將輸入數據拆分為兩部分：
        # amount 分支：只取 "amount" 欄位（已正規化）
        input_cols_amount = ['amount']
        # 其他分支：取其他特徵，這裡排除 money
        input_cols_other = ['open', 'max', 'min', 'close', 'delta', 'deal']
        
        amount_val = self.data.iloc[idx][input_cols_amount].values.astype('float32')
        other_vals = self.data.iloc[idx][input_cols_other].apply(
            lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x
        ).values.astype('float32')
        
        # 將輸入分支分別返回，後續模型將分別處理
        features = {
            'amount': torch.tensor(amount_val),
            'other': torch.tensor(other_vals)
        }
        # 目標：預測下一日的 [amount, 'open', max, min, close]
        target_cols = ['amount', 'max', 'min', 'close']
        target = self.data.iloc[idx + 1][target_cols].apply(
            lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x
        ).values.astype('float32')
        
        return features, torch.tensor(target)


# 2. 定義多分支神經網路模型
class MultiBranchStockPredictor(nn.Module):
    def __init__(self, other_input_size=6, hidden_size=32, output_size=4):
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
        # 整合兩分支後接全連接層，輸出 4 維 (amount, max, min, close)
        self.combined_fc = nn.Sequential(
            nn.Linear(16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, features):
        # features 為字典：{'amount': ..., 'other': ...}
        amount_input = features['amount']  # shape: (batch, 1)
        other_input = features['other']    # shape: (batch, 6)
        
        out_amount = self.amount_branch(amount_input)
        out_other = self.other_branch(other_input)
        
        # 連接兩分支的輸出
        combined = torch.cat((out_amount, out_other), dim=1)
        output = self.combined_fc(combined)
        return output


if __name__ == '__main__':
    csv_file = r'D:\TradePredictor\data\STOCK_DAY_2308.csv'
    dataset = StockDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiBranchStockPredictor(other_input_size=6, hidden_size=64, output_size=4)
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        scheduler.step()  # 更新學習率

        if avg_loss < threshold:
            print(f"Loss has approached {threshold}, stopping training early.")
            break

    # 以最新一天的資料預測明日目標值
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
    target_cols = ['預測明日amount', '預測明日max', '預測明日min', '預測明日close']
    for col, val in zip(target_cols, pred_np):
        print(f"{col}: {val:.2f}")
