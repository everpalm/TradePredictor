# Content of tests/test_steel/conftest.py
'''Copyright (c) 2025 Jaron Cheng'''
import pytest
from steel.product import IntegratedFactory
from torch.utils.data import DataLoader
from generic.data_modeling import MultiBranchStockPredictor
from generic.data_modeling import StockDataset
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


@pytest.fixture(scope="package")
def integrated_steel():
    '''docstring'''
    csv_file = r'D:\TradePredictor\data\STOCK_DAY_2002.csv'

    print('\n\033[32m============== Setup Integrated Steel ==========\033[0m')
    csv_model = MultiBranchStockPredictor(
        other_input_size=6, hidden_size=64, output_size=6
    )
    csv_dataset = StockDataset(csv_file)
    csv_optimizer = Adam(csv_model.parameters(), lr=0.001)

    stock = IntegratedFactory("2002")
    return stock.create_predictor(
        model=csv_model,
        dataset=csv_dataset,
        dataloader=DataLoader(csv_dataset, batch_size=32, shuffle=True),
        criterion=MSELoss(),
        optimizer=csv_optimizer,
        scheduler=StepLR(csv_optimizer, step_size=10, gamma=0.95)
    )