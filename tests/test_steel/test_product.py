# Content of tests/test_product.py
'''Copyright (c) 2025 Jaron Cheng'''
import logging

logger = logging.getLogger(__name__)


class TestIntegrated:
    """
    Test suite for the Integrated steel product predictor.

    This class contains test cases that validate the functionality
    of the Integrated predictor for steel products. It ensures that
    predictions are within reasonable bounds and maintain expected
    relationships between different price metrics.
    """
    def test_integrated_steel(self, integrated_steel):
        """
        Test the integrated steel product prediction functionality.

        This test verifies that the integrated_steel predictor generates
        reasonable predictions for trading metrics. It checks that:
        1. The maximum price is not more than 10% above the opening price
        2. The minimum price is not more than 10% below the opening price
        3. The average price is between the minimum and maximum prices

        Parameters
        ----------
        integrated_steel : BasePredictor
            A fixture providing an instance of the integrated steel
            product predictor.

        Returns
        -------
        None
            This test function does not return a value but raises an
            AssertionError if any of the assertions fail.
        """
        predict: list = integrated_steel.predict_trade(1000, 0.75)
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
