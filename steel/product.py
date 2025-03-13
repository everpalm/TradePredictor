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

logger = get_logger(__name__, logging.DEBUG)


class BasePredictor(ABC):
    '''docstring'''
    def __init__(self,
            model: MultiBranchStockPredictor,
            dataset: StockDataset,
            dataloader: DataLoader,
            criterion: MSELoss,
            optimizer: Adam,
            scheduler: StepLR,
        ):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    @abstractmethod
    def predict_trade(self):
        '''docstring'''
        pass


class Integrated(BasePredictor):
    '''docstring'''
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
                logger.info("Loss has approached %d, stopping training early.", threshold)
                break

        # 以最新一天的資料預測明日目標值
        last_row = self.dataset.data.iloc[-1]
        input_cols_amount = ['amount']
        input_cols_other = ['open', 'avg', 'max', 'min', 'close', 'deal']
        amount_val = last_row[input_cols_amount].values.astype('float32')
        other_vals = last_row[input_cols_other].apply(
            lambda x: float(str(x).replace(',', '')) if isinstance(x, str) else x
        ).values.astype('float32')

        features = {
            'amount': torch.tensor(amount_val).unsqueeze(0),  # 增加 batch 維度
            'other': torch.tensor(other_vals).unsqueeze(0)
        }

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features)

        pred_np = prediction.numpy().flatten()
        target_cols = ['預測明日amount', '預測明日open', '預測明日avg', '預測明日max', '預測明日min', '預測明日close']
        
        # 將預測的 normalized amount（pred_np[0]）轉換回原始尺度
        normalized_amount = pred_np[0]
        original_amount = self.dataset.scaler_amount.inverse_transform([[normalized_amount]])[0][0]
        logger.info("預測明日amount(原始尺度: %d", int(original_amount/1000))
        
        # 輸出其餘預測值
        for col, val in zip(target_cols[1:], pred_np[1:]):
            logger.info(f"{col}: {val:.2f}")


class BaseSteelFactory(ABC):
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


class IntegratedFactory(BaseSteelFactory):
    def create_predictor(self, **kwargs) -> BasePredictor:
        if self.code == '2002':
            return Integrated(**kwargs)
        else:
            raise ValueError(f"Unsupported Stock Code: {self.code}")