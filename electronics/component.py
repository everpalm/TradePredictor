# Contents of electronics/component.py
'''Copyright (c) 2025 Jaron Cheng'''
import logging

from generic.predictor import BasePredictor
from generic.predictor import BasePredictorFactory
from unit.log_handler import get_logger

logger = get_logger(__name__, logging.INFO)


class Peripheral(BasePredictor):
    '''docstring'''


class PeripheralFactory(BasePredictorFactory):
    def create_predictor(self, **kwargs) -> BasePredictor:
        if self.code == '2308':
            return Peripheral(**kwargs)
        else:
            raise ValueError(f"Unsupported Stock Code: {self.code}")