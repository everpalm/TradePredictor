# Content of tests/test_component.py
'''Copyright (c) 2025 Jaron Cheng'''
import logging

logger = logging.getLogger(__name__)


class TestPeripheral:
    """
    Test suite for the Peripheral electronics component predictor.

    This class contains test cases that validate the functionality
    of the Peripheral predictor for electronics components. It checks
    that predictions are within reasonable bounds and maintain
    expected relationships between different price values.
    """
    def test_electro_component(self, component_electronics):
        """
        Test the electronic component prediction functionality.

        This test verifies that the component_electronics predictor generates
        reasonable predictions for trading metrics. It checks that:
        1. The maximum price is not more than 10% above the opening price
        2. The minimum price is not more than 10% below the opening price
        3. The average price is between the minimum and maximum prices

        Parameters
        ----------
        component_electronics : BasePredictor
            A fixture providing an instance of the electronics component
            predictor.

        Returns
        -------
        None
            This test function does not return a value but raises an
            AssertionError if any of the assertions fail.
        """
        predict: list = component_electronics.predict_trade(1000, 39)
        amount: int = predict[0]
        open: float = predict[1]
        avg: float = predict[2]
        max: float = predict[3]
        min: float = predict[4]
        close: float = predict[5]
        logger.info("amount = %d", amount)
        logger.info("open = %.2f", open)
        logger.info("avg = %.2f", avg)
        logger.info("max = %.2f", max)
        logger.info("min = %.2f", min)
        logger.info("close = %.2f", close)
        assert max <= open * 1.1
        assert min >= open * 0.9
        assert max >= avg >= min
