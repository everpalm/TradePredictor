import logging
# import torch

from generic.predictor import BasePredictor
from generic.predictor import BasePredictorFactory
from unit.log_handler import get_logger

logger = get_logger(__name__, logging.INFO)


class Integrated(BasePredictor):
    '''docstring'''


class IntegratedFactory(BasePredictorFactory):
    def create_predictor(self, **kwargs) -> BasePredictor:
        if self.code == '2002':
            return Integrated(**kwargs)
        else:
            raise ValueError(f"Unsupported Stock Code: {self.code}")