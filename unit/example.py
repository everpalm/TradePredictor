import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 定義資料集 (範例與之前類似)
class StockDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, encoding='big5')
        self.data.columns = [col.strip() for col in self.data.columns]
        if 'date' in self.data.columns:
            self.data = self.data.sort_values(by='date').reset_index(drop=True)
        print("CSV 欄位名稱：", self.data.columns.tolist())
        
        # 對 "amount" 與 "deal" 分別處理：
        # amount：移除逗號後正規化
        self.data['amount'] = self.data['amount'].apply(lambda x: float(str(x).replace(',', '')))
        scaler_amount = MinMaxScaler(feature_range=(0, 100))
        self.data[['amount']] = scaler_amount.fit_transform(self.data[['amount']])
        
        # 對其他欄位若有需要也可以處理（此處只處理 amount）
        # 其他欄位保留原始數值，這裡假設其他欄位不含逗號或已清理

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 將輸入分為兩部分：
        # amount (單獨處理) 與其他特徵：open, max, min, close, delta, deal
        input_cols_amount = ['amount']
        input_cols_other = ['open', 'max', 'min', 'close', 'delta', 'deal']
        
        # amount 欄位已經正規化
        amount_val = self.data.iloc[idx][input_cols_amount].values.astype('float32')
        # 其他欄位：如果有逗號就先移除（依據實際情況）
        other_vals = self.data.iloc[idx][input_cols_other].apply(
            lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x
        ).values.astype('float32')
        
        # 合併後形成完整輸入，例如也可以讓模型各分支獨立輸入（這裡示範分支模型）
        # 我們這裡返回兩個部分，模型可以分別處理
        features = {
            'amount': torch.tensor(amount_val),
            'other': torch.tensor(other_vals)
        }
        # 目標：預測下一日的 max, min, close
        target_cols = ['max', 'min', 'close']
        target = self.data.iloc[idx + 1][target_cols].apply(
            lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x
        ).values.astype('float32')
        
        return features, torch.tensor(target)


# 定義多分支神經網路模型
class MultiBranchStockPredictor(nn.Module):
    def __init__(self, other_input_size=6, hidden_size=32, output_size=3):
        super(MultiBranchStockPredictor, self).__init__()
        # 分支1：處理 amount (單一特徵)
        self.amount_branch = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU()
        )
        # 分支2：處理其他特徵 (6 個特徵)
        self.other_branch = nn.Sequential(
            nn.Linear(other_input_size, 8),
            nn.ReLU()
        )
        # 整合後的全連接層
        self.combined_fc = nn.Sequential(
            nn.Linear(16, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, features):
        # features 是一個字典：{'amount': ..., 'other': ...}
        amount_input = features['amount']  # shape: (batch, 1)
        other_input = features['other']    # shape: (batch, 6)
        
        out_amount = self.amount_branch(amount_input)
        out_other = self.other_branch(other_input)
        
        # 連接兩個分支的輸出
        combined = torch.cat((out_amount, out_other), dim=1)
        output = self.combined_fc(combined)
        return output


if __name__ == '__main__':
    csv_file = r'D:\TradePredictor\data\STOCK_DAY_2002_202503.csv'
    dataset = StockDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MultiBranchStockPredictor(other_input_size=6, hidden_size=32, output_size=3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

    # 預測：以最新一天的資料進行預測
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
    target_cols = ['預測明日max', '預測明日min', '預測明日close']
    for col, val in zip(target_cols, pred_np):
        print(f"{col}: {val:.2f}")
