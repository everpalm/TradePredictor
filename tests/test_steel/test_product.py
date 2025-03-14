# Content of tests/test_product.py
'''Copyright (c) 2025 Jaron Cheng'''
import logging

logger = logging.getLogger(__name__)


class TestIntegrated:
    '''docstring'''
    def test_integrated_steel(self, integrated_steel):
        predict: list = integrated_steel.predict_trade(500, 0.9)
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