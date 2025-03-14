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
def steel_model():
    '''docstring'''
    print('\n\033[32m============== Setup Steel Model ===============\033[0m')
    csv_model = MultiBranchStockPredictor(
        other_input_size=6, hidden_size=32, output_size=6
    )
    return csv_model 


@pytest.fixture(scope="package")
def steel_dataset():
    print('\n\033[32m============== Setup Steel Dataset =============\033[0m')
    csv_file = r'D:\TradePredictor\data\STOCK_DAY_2002.csv'
    csv_dataset = StockDataset(csv_file)
    return csv_dataset


@pytest.fixture(scope="package")
def steel_optimizer(steel_model):
    print('\n\033[32m============== Setup Steel Optimizer ===========\033[0m')
    csv_optimizer = Adam(steel_model.parameters(), lr=0.001)
    return csv_optimizer


@pytest.fixture(scope="package")
def integrated_steel(steel_model, steel_dataset, steel_optimizer):
    '''docstring'''
    print('\n\033[32m============== Setup Integrated Steel ==========\033[0m')
    stock = IntegratedFactory("2002")
    return stock.create_predictor(
        model=steel_model,
        dataset=steel_dataset,
        dataloader=DataLoader(steel_dataset, batch_size=32, shuffle=True),
        criterion=MSELoss(),
        optimizer=steel_optimizer,
        scheduler=StepLR(steel_optimizer, step_size=10, gamma=0.95)
    )