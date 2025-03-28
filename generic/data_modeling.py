# Contents of unit/data_modeling.py
'''Copyright (c) 2025 Jaron Cheng'''

import logging
import pandas as pd
import torch
import torch.nn as nn

from unit.log_handler import get_logger
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

logger = get_logger(__name__, logging.INFO)


# 1. 定義資料集
class StockDataset(Dataset):
    """
    A PyTorch Dataset for stock market data processing and preparation.

    This class handles loading stock data from CSV files, cleaning and
    preprocessing the data, and providing structured access for model training.
    It performs several
    key preprocessing steps:
    1. Encoding handling for Chinese characters (Big5)
    2. Date-based sorting
    3. Numeric conversions (removing commas from financial figures)
    4. Feature engineering (calculating average price)
    5. Feature normalization using MinMaxScaler

    Parameters
    ----------
    csv_file : str
        Path to the CSV file containing stock market data.

    Attributes
    ----------
    data : pandas.DataFrame
        The processed stock data.
    scaler_amount : sklearn.preprocessing.MinMaxScaler
        Scaler used for normalizing the 'amount' column.
    """
    def __init__(self, csv_file):
        # 使用 Big5 編碼讀取 CSV（根據實際編碼調整）
        self.data = pd.read_csv(csv_file, encoding='big5')

        # 清理欄位名稱，移除前後空格
        self.data.columns = [col.strip() for col in self.data.columns]

        # 依日期排序（若有 date 欄位）
        if 'date' in self.data.columns:
            self.data = self.data.sort_values(by='date').reset_index(drop=True)
        logger.debug("CSV 欄位名稱： %s", self.data.columns.tolist())

        # 將 "amount" 與 "money" 欄位轉換成 float（移除逗號）
        self.data['amount'] = self.data['amount'].apply(
            lambda x: float(str(x).replace(',', '')))
        self.data['money'] = self.data['money'].apply(
            lambda x: float(str(x).replace(',', '')))
        # 保存原始 amount 值用於計算 avg
        self.data['amount_raw'] = self.data['amount']
        # 計算平均股價 avg = money / amount_raw
        self.data['avg'] = self.data['money'] / self.data['amount_raw']

        # 對 "amount" 欄位進行正規化，使用 feature_range=(0, 10)
        scaler_amount = MinMaxScaler(feature_range=(0, 10))
        self.data[['amount']] = scaler_amount.fit_transform(
            self.data[['amount']])
        # 保存 scaler_amount 方便後續逆轉換
        self.scaler_amount = scaler_amount

        # 對 "deal" 欄位進行正規化（feature_range=(0, 1)）
        self.data['deal'] = self.data['deal'].apply(
            lambda x: float(str(x).replace(',', '')))
        scaler_deal = MinMaxScaler(feature_range=(0, 1))
        self.data[['deal']] = scaler_deal.fit_transform(self.data[['deal']])

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 將輸入數據拆分為兩部分：
        # 分支1: 處理 amount（已正規化）
        input_cols_amount = ['amount']
        # 分支2: 其他特徵，移除 'delta'
        input_cols_other = ['open', 'avg', 'max', 'min', 'close', 'deal']

        amount_val = self.data.iloc[idx][input_cols_amount].values.astype(
            'float32')
        other_vals = self.data.iloc[idx][input_cols_other].apply(
            lambda x: (
                float(str(x).replace(',', '')) if isinstance(x, str) else x
            )
        ).values.astype('float32')

        features = {
            'amount': torch.tensor(amount_val),
            'other': torch.tensor(other_vals)
        }
        # 目標：預測下一日的 [amount, open, avg, max, min, close]
        target_cols = ['amount', 'open', 'avg', 'max', 'min', 'close']
        target = self.data.iloc[idx + 1][target_cols].apply(
            lambda x: (
                float(str(x).replace(',', '')) if isinstance(x, str) else x
            )
        ).values.astype('float32')

        return features, torch.tensor(target)


# 2. 定義多分支神經網路模型
class MultiBranchStockPredictor(nn.Module):
    """
    A multi-branch neural network for stock price prediction.

    This model uses a specialized architecture with two separate branches:
    1. A branch dedicated to processing trading volume (amount)
    2. A branch for other stock metrics (open, avg, max, min, close, deal)

    The outputs from both branches are concatenated and passed through
    additional fully connected layers to produce the final predictions.

    Parameters
    ----------
    other_input_size : int, default=6
        The number of features in the 'other' branch (excluding 'amount').
    hidden_size : int, default=32
        The size of the hidden layer in the combined network.
    output_size : int, default=6
        The number of target variables to predict.

    Attributes
    ----------
    amount_branch : nn.Sequential
        Neural network branch for processing the 'amount' feature.
    other_branch : nn.Sequential
        Neural network branch for processing other stock metrics.
    combined_fc : nn.Sequential
        Fully connected layers after combining both branches.
    """
    def __init__(self, other_input_size=6, hidden_size=32, output_size=6):
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
        # 整合兩分支後接全連接層，輸出 6 維 (amount, open, avg, max, min, close)
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
