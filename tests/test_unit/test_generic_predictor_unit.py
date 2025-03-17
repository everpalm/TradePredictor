import numpy as np
import pandas as pd
# import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
from torch.utils.data import DataLoader, Dataset

# 從 generic/predictor.py 載入 BasePredictor
from generic.predictor import BasePredictor

# -------------------------------
# Dummy implementations for testing
# -------------------------------


class DummyModel(nn.Module):
    """
    Dummy model for testing purposes.

    This dummy model includes a dummy parameter so that model.parameters() is
    non-empty.
    Its forward method returns a fixed tensor of shape (1,6) representing the
    predictions
    in the order:
        [normalized_amount, open, avg, max, min, close]

    The output is computed as:
        output = dummy_param * base_output
    where base_output is fixed.
    """
    def __init__(self):
        super().__init__()
        # Add a dummy parameter to ensure there is at least one parameter.
        self.dummy_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, features):
        # Fixed base output
        base_output = torch.tensor(
            [[0, 100, 10, 110, 90, 105]],
            dtype=torch.float32,
            device=self.dummy_param.device
        )
        # Multiply by dummy_param to ensure the output requires grad.
        return base_output * self.dummy_param


class DummyDataset(Dataset):
    """
    Dummy dataset for testing.

    Construct a simple DataFrame with two rows and the following columns:
        'amount', 'money', 'open', 'max', 'min', 'close', 'delta', 'deal'
    Compute avg = money / amount (using the raw amount value).
    'amount' is normalized using MinMaxScaler with feature_range=(0,10),
    and 'deal' is normalized with feature_range=(0,1).
    """
    def __init__(self):
        data = {
            'amount': [100, 200],   # 原始數值
            'money': [1000, 2000],  # avg = 1000/100 = 10, 2000/200 = 10
            'open': [100, 150],
            'max': [110, 160],
            'min': [90, 140],
            'close': [105, 155],
            'delta': [0.5, 0.5],
            'deal': [100, 200]
        }
        self.data = pd.DataFrame(data)
        # 保存原始 amount 值以便計算 avg
        self.data['amount_raw'] = self.data['amount']
        # 計算 avg
        self.data['avg'] = self.data['money'] / self.data['amount_raw']

        # Normalize 'amount' using feature_range (0,10)
        from sklearn.preprocessing import MinMaxScaler
        scaler_amount = MinMaxScaler(feature_range=(0, 10))
        self.data[['amount']] = scaler_amount.fit_transform(
            self.data[['amount']])
        self.scaler_amount = scaler_amount

        # Normalize 'deal' using feature_range (0,1)
        from sklearn.preprocessing import MinMaxScaler as MMS
        scaler_deal = MMS(feature_range=(0, 1))
        self.data['deal'] = self.data['deal'].astype(float)
        self.data[['deal']] = scaler_deal.fit_transform(self.data[['deal']])

    def __len__(self):
        # __getitem__ uses idx+1 as target, so length is len(data)-1
        return len(self.data) - 1

    def __getitem__(self, idx):
        # 將輸入數據拆分為兩部分：
        # 分支1: 'amount'
        # 分支2: ['open', 'avg', 'max', 'min', 'close', 'delta', 'deal']
        input_cols_amount = ['amount']
        input_cols_other = ['open', 'avg', 'max', 'min', 'close', 'delta',
                            'deal']
        amount_val = self.data.iloc[idx][input_cols_amount].values.astype(
            'float32')
        other_vals = self.data.iloc[idx][input_cols_other].values.astype(
            'float32')
        features = {
            'amount': torch.tensor(amount_val),
            'other': torch.tensor(other_vals)
        }
        # 目標: 預測下一日的 [amount, open, avg, max, min, close]
        target_cols = ['amount', 'open', 'avg', 'max', 'min', 'close']
        target = self.data.iloc[idx + 1][target_cols].values.astype('float32')
        return features, torch.tensor(target)


# DummyPredictor 繼承自 BasePredictor，不需覆寫 predict_trade
class DummyPredictor(BasePredictor):
    pass

# -------------------------------
# 測試函數
# -------------------------------


def test_adjust_to_tick_size():
    """
    測試 BasePredictor.adjust_to_tick_size 方法，
    檢查其是否根據不同價格返回正確調整後的價格。
    """
    # 這裡直接使用內部計算結果
    assert BasePredictor.adjust_to_tick_size(9.03) == round(
        round(9.03 / 0.01) * 0.01, 2)
    assert BasePredictor.adjust_to_tick_size(25.12) == round(
        round(25.12 / 0.05) * 0.05, 2)
    assert BasePredictor.adjust_to_tick_size(75.7) == round(
        round(75.7 / 0.1) * 0.1, 2)
    assert BasePredictor.adjust_to_tick_size(300) == round(
        round(300 / 0.5) * 0.5, 0)
    assert BasePredictor.adjust_to_tick_size(800) == round(
        round(800 / 1) * 1, 0)
    assert BasePredictor.adjust_to_tick_size(1500) == round(
        round(1500 / 5) * 5, 0)


def test_predict_trade():
    """
    測試 BasePredictor.predict_trade 方法。

    使用 DummyModel、DummyDataset 建立一個 DummyPredictor,
    呼叫 predict_trade 並檢查返回的預測值是否符合預期。

    預期：
    - DummyModel 固定返回 (經 dummy_param 影響後)
    tensor([[0, 100, 10, 110, 90, 105]])。
      由於 dummy_param 初始為 1.0, 結果應與固定值相同。
    - 在 predict_trade 中，第一個值 (normalized amount = 0)
    經過逆轉換後應還原到 DummyDataset 的原始 amount,
      然後除以 1000 取整後設為 0 (根據 dummy 數據)。
    - 其餘價格經過 tick size 調整後保持不變。
    最終預期輸出為:
        [0, 100, 10, 110, 90, 105]
    """
    # 建立 DummyModel 與 DummyDataset
    dummy_model = DummyModel()
    dummy_dataset = DummyDataset()
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)

    # 建立 optimizer, criterion, scheduler
    dummy_optimizer = Adam(dummy_model.parameters(), lr=0.001)
    dummy_criterion = MSELoss()
    dummy_scheduler = StepLR(dummy_optimizer, step_size=10, gamma=0.95)

    # 建立 DummyPredictor 實例
    predictor = DummyPredictor(
        model=dummy_model,
        dataset=dummy_dataset,
        dataloader=dummy_dataloader,
        criterion=dummy_criterion,
        optimizer=dummy_optimizer,
        scheduler=dummy_scheduler
    )

    # 執行 predict_trade，設定 num_epochs=1 並 threshold 為一個較高的值以避免提前停止
    pred = predictor.predict_trade(num_epochs=1, threshold=1000)

    # 預期輸出：由於 DummyModel 返回固定輸出：
    # [normalized_amount, open, avg, max, min, close] = [0, 100, 10, 110, 90,
    # 105]
    # 在 predict_trade 中，第一個值 (normalized amount = 0) 被逆轉換：
    # 原始 amount = scaler_amount.inverse_transform([[0]]) -> 根據 DummyDataset 原始
    # amount = 100，
    # 然後除以 1000取整，得到 0。
    # 其他價格則經 tick size 調整後保持不變。
    expected = np.array([0, 100, 10, 110, 90, 105], dtype=np.float32)
    # 使用 assert_allclose 並設定容忍度，以處理浮點數精度誤差
    np.testing.assert_allclose(pred, expected, atol=0.1)
