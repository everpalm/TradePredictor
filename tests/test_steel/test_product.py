# Content of tests/test_product.py
'''Copyright (c) 2025 Jaron Cheng'''
import logging

logger = logging.getLogger(__name__)


class TestIntegrated:
    '''docstring'''
    def test_integrated_steel(self, integrated_steel):
        integrated_steel.predict_trade(1000, 0.72)