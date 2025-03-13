# Contents of generic/predictor.py
'''Copyright (c) 2025 Jaron Cheng'''
from abc import ABC
from abc import abstractmethod
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from unit.log_handler import get_logger
from generic.data_modeling import MultiBranchStockPredictor
from generic.data_modeling import StockDataset


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
    def predict_trade(self, num_epochs: int, threshold: float):
        pass


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