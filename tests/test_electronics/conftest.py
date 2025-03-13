# Content of tests/test_electronics/conftest.py
'''Copyright (c) 2025 Jaron Cheng'''
import pytest
from electronics.component import PeripheralFactory
from torch.utils.data import DataLoader
from generic.data_modeling import MultiBranchStockPredictor
from generic.data_modeling import StockDataset
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR


@pytest.fixture(scope="package")
def electro_model():
    '''docstring'''
    print('\n\033[32m============== Setup Steel Model ===============\033[0m')
    model = MultiBranchStockPredictor(
        other_input_size=6, hidden_size=64, output_size=6
    )
    return model 


@pytest.fixture(scope="package")
def electro_dataset():
    print('\n\033[32m============== Setup Steel Dataset =============\033[0m')
    file = r'D:\TradePredictor\data\STOCK_DAY_2308.csv'
    dataset = StockDataset(file)
    return dataset


@pytest.fixture(scope="package")
def electro_optimizer(electro_model):
    print('\n\033[32m============== Setup Steel Optimizer ===========\033[0m')
    optimizer = Adam(electro_model.parameters(), lr=0.001)
    return optimizer


@pytest.fixture(scope="package")
def component_electronics(electro_model, electro_dataset, electro_optimizer):
    '''docstring'''
    print('\n\033[32m============== Setup Integrated Steel ==========\033[0m')
    stock = PeripheralFactory("2308")
    return stock.create_predictor(
        model=electro_model,
        dataset=electro_dataset,
        dataloader=DataLoader(electro_dataset, batch_size=32, shuffle=True),
        criterion=MSELoss(),
        optimizer=electro_optimizer,
        scheduler=StepLR(electro_optimizer, step_size=10, gamma=0.95)
    )