# Contents of generic/predictor.py
'''Copyright (c) 2025 Jaron Cheng'''
import logging
import torch

from abc import ABC
from abc import abstractmethod
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from unit.log_handler import get_logger
from generic.data_modeling import MultiBranchStockPredictor
from generic.data_modeling import StockDataset

logger = get_logger(__name__, logging.INFO)


class BasePredictor(ABC):
    '''docstring'''
    def __init__(
            self,
            model: MultiBranchStockPredictor,
            dataset: StockDataset,
            dataloader: DataLoader,
            criterion: MSELoss,
            optimizer: Adam,
            scheduler: StepLR):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    # @abstractmethod
    # def predict_trade(self, num_epochs: int, threshold: float):
    #     pass

    def adjust_to_tick_size(self, price):
        """根據台股檔位規則調整價格到合法值"""
        if price < 10:
            tick_size = 0.01
        elif price < 50:
            tick_size = 0.05
        elif price < 100:
            tick_size = 0.1
        elif price < 500:
            tick_size = 0.5
        elif price < 1000:
            tick_size = 1
        else:
            tick_size = 5

        # 調整到最接近的檔位值（四捨五入到 tick_size 的倍數）
        adjusted_price = round(price / tick_size) * tick_size
        # 確保小數位數符合規則（避免浮點數精度問題）
        return round(adjusted_price, 2 if tick_size < 1 else 0)

    def predict_trade(self, num_epochs: int, threshold: float):
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for features, targets in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.dataloader)

            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            self.scheduler.step()

            if avg_loss < threshold:
                logger.debug(
                    "Loss has approached %d, stopping training early.",
                    threshold
                )
                break

        # 以最新一天的資料預測明日目標值
        last_row = self.dataset.data.iloc[-1]
        input_cols_amount = ['amount']
        input_cols_other = ['open', 'avg', 'max', 'min', 'close', 'deal']
        amount_val = last_row[input_cols_amount].values.astype('float32')
        other_vals = last_row[input_cols_other].apply(
            lambda x: (
                float(str(x).replace(',', '')) if isinstance(x, str) else x
            )
        ).values.astype('float32')

        features = {
            'amount': torch.tensor(amount_val).unsqueeze(0),  # 增加 batch 維度
            'other': torch.tensor(other_vals).unsqueeze(0)
        }

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features)

        pred_np = prediction.numpy().flatten()
        target_cols = [
            '預測明日amount',
            '預測明日open',
            '預測明日avg',
            '預測明日max',
            '預測明日min',
            '預測明日close'
        ]

        # 將預測的 normalized amount（pred_np[0]）轉換回原始尺度
        normalized_amount = pred_np[0]
        original_amount = self.dataset.scaler_amount.inverse_transform(
            [[normalized_amount]])[0][0]
        logger.debug("預測明日amount(原始尺度: %d)", int(original_amount/1000))
        pred_np[0] = int(original_amount/1000)

        # 對 open, max, min, close 應用檔位限制，avg 不調整
        price_indices = [1, 3, 4, 5]  # open, max, min, close 的索引

        # 對股價相關預測值應用檔位限制
        for i in price_indices:
            pred_np[i] = self.adjust_to_tick_size(pred_np[i])

        # 輸出其餘預測值
        for col, val in zip(target_cols[1:], pred_np[1:]):
            logger.debug(f"{col}: {val:.2f}")

        return pred_np


class BasePredictorFactory(ABC):
    def __init__(self, code: str):
        """
        Initialize the BasePlatformFactory with an API instance.

        Args:
            api (BaseInterface): An interface instance providing platform
            details.
        """
        self.code = code

    @abstractmethod
    def create_predictor(self) -> BasePredictor:
        pass
