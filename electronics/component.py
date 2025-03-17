# Contents of electronics/component.py
'''Copyright (c) 2025 Jaron Cheng'''
import logging

from generic.predictor import BasePredictor
from generic.predictor import BasePredictorFactory
from unit.log_handler import get_logger

logger = get_logger(__name__, logging.INFO)


class Peripheral(BasePredictor):
    """
    Peripheral component stock predictor.

    This class implements the BasePredictor interface to provide
    specialized prediction functionality for electronics peripheral
    component stocks. It focuses on analyzing and forecasting
    trends specific to peripheral electronic components.

    Inherits from BasePredictor to maintain a consistent interface
    for all prediction implementations.
    """


class PeripheralFactory(BasePredictorFactory):
    """
    Factory class for creating Peripheral predictor instances.

    This factory creates appropriate Peripheral predictor objects
    based on stock code and other parameters. Currently supports
    only stock code '2308'.

    Inherits from BasePredictorFactory to maintain a consistent
    factory pattern implementation across the system.
    """
    def create_predictor(self, **kwargs) -> BasePredictor:
        """
        Create and return an appropriate peripheral predictor instance.

        Based on the stock code configured in the factory, this method
        returns the appropriate predictor implementation. Currently
        only supports stock code '2308'.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the predictor constructor.

        Returns
        -------
        BasePredictor
            An instance of a class implementing the BasePredictor interface.

        Raises
        ------
        ValueError
            If the requested stock code is not supported by this factory.
        """
        if self.code == '2308':
            return Peripheral(**kwargs)
        else:
            raise ValueError(f"Unsupported Stock Code: {self.code}")
