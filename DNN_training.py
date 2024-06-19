import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from DNN import MaxoutNetWithRegularization
from DNN_funcs import *

import numpy as np
import pandas as pd
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


# Global Variables
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

Feature_Path = "./data/model_data/period_features"
Save_Path = "./result/nn"
Model_Path = "./data/model_data/models/nn"
os.makedirs(Save_Path, exist_ok=True)
os.makedirs(Model_Path, exist_ok=True)

Period_Lst = list(range(25))

BATCHSIZE = 512
EPOCHS = 400
LEARNING_RATE = 0.001
EARLYSTOPPING_PATIENCE = 20

# Load data
def load_feature(num_period: int, feature_path: str = Feature_Path) -> pd.DataFrame:
    data = pd.read_parquet(f"{feature_path}/features_period_{num_period}.parquet")

    date_range = data["Date"].sort_values().unique()
    data = data.set_index("Date")

    train_data = data.loc[date_range[:-250]]
    test_data = data.loc[date_range[-250:]]

    return train_data.reset_index(), test_data.reset_index()


# generate train and test data
def generate_train_test_data(period: int):
    train_data, test_data = load_feature(period)
    
    X_train = train_data.drop(columns=["Date", "Ticker", "Return_tomorrow", "Target"])
    y_train = train_data["Target"]
    X_test = test_data.drop(columns=["Date", "Ticker", "Return_tomorrow", "Target"])
    y_test = test_data["Target"]

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    X_test = torch.tensor(X_test.values, dtype=torch.float32, requires_grad=False)
    y_test = torch.tensor(y_test.values, dtype=torch.long, requires_grad=False)

    stock_dataset = StockDataset(X_train, y_train)
    train_dataloader = DataLoader(stock_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4)

    return train_dataloader, X_test, y_test, test_data


# train a period
def train_a_period(train_dataloader, device=device):
    model = MaxoutNetWithRegularization().to(device)

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    early_stopping = EarlyStopping(patience=EARLYSTOPPING_PATIENCE, verbose=True, delta=0.0001)

    losses = []
    gradients = []

    model.train()

    for epoch in range(EPOCHS):
        epoch_losses = []
        epoch_gradients = []
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')

        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            loss = criteria(outputs, targets) + model.l1_regularization()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_gradients.append(model.maxout1.linear_layers[0].weight.grad.norm().item())

            progress_bar.set_postfix(loss=np.round(np.mean(epoch_losses), 4), gradient=np.round(np.mean(epoch_gradients), 4))

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_gradient = sum(epoch_gradients) / len(epoch_gradients)
        losses.append(avg_loss)
        gradients.append(avg_gradient)

        early_stopping(avg_loss)
        if early_stopping.early_stop:
            break

    return model, losses, gradients


# predict a period
def summary_model_performance(period, X_test, y_test, model, losses, gradients):
    fig = plot_losses_gradients(losses, gradients, period)
    
    probabilities, predicted_labels, performance_stats = predict(model, X_test, y_test, nn.CrossEntropyLoss())

    return probabilities, predicted_labels, performance_stats


if __name__ == "__main__":
    start_time = time.time()
    performance_stats_sum = {}

    for period in Period_Lst:
        print("="*50)
        print(f"Processing period {period}")
        print("="*50)

        train_dataloader, X_test, y_test, ori_test_data = generate_train_test_data(period)

        model, losses, gradients = train_a_period(train_dataloader)

        probabilities, predicted_labels, performance_stats = summary_model_performance(period, X_test, y_test, model, losses, gradients)

        performance_stats_sum[period] = performance_stats
        print(performance_stats)
        
        ori_test_data = ori_test_data[["Date", "Ticker", "Return_tomorrow", "Target"]].copy()
        ori_test_data["pred"] = predicted_labels
        ori_test_data["pred_proba"] = probabilities

        ori_test_data.to_parquet(f"{Save_Path}/nn_period_{period}.parquet", index=False)
        
        torch.save(model.state_dict(), f"{Model_Path}/nn_period_{period}.pt")
    
    performance_stats_df = pd.DataFrame(performance_stats_sum)
    performance_stats_df.to_csv(f"{Save_Path}/nn_performance_stats.csv")

    all__df = pd.concat([pd.read_parquet(f"{Save_Path}/nn_period_{period}.parquet") for period in Period_Lst])
    all__df.to_parquet(f"{Save_Path}/nn_all_periods.parquet", index=False)

    passed_time = time.time() - start_time
    print("--- %s min %s s ---" % (passed_time // 60, round(passed_time % 60)))