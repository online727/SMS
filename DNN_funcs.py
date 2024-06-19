import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

class StockDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def generate_dataloader(features, labels, batch_size=256, shuffle=True):
    dataset = StockDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0.0001):
        """
        Args:
            patience (int): the number of epochs to wait before early stopping
            verbose (bool): if True, prints a message when early stopping is triggered
            delta (float): the loss value that is considered significant
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, train_loss):
        if self.best_loss is None:
            self.best_loss = train_loss
        elif train_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
        else:
            self.best_loss = train_loss
            self.counter = 0


def plot_losses_gradients(train_losses, train_grads, period):
    # 绘制损失曲线
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制梯度曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_grads, label='Gradient Norm', color='red')
    plt.title('Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()

    plt.suptitle(f'Training Loss and Gradient Norm of period {period}')

    plt.tight_layout()
    plt.savefig(f"./result/nn/period_{period}_losses_gradients.png")
    plt.close("all")
    return fig


def predict(model, X_test, y_test, criteria):
    """
    model: torch.nn.Module
    X_test: torch.Tensor, require_grad=False
    y_test: torch.Tensor, require_grad=False
    criteria: loss function, cross entropy loss
    """
    model.eval()
    model = model.to("cpu")

    with torch.no_grad():
        outputs = model(X_test)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        test_loss = criteria(outputs, y_test)
        accuracy = accuracy_score(y_test, predicted_labels)
        recall = recall_score(y_test, predicted_labels)
        precision = precision_score(y_test, predicted_labels)
        f1 = f1_score(y_test, predicted_labels)
        auc = roc_auc_score(y_test, probabilities[:, 1])
        gini_index = 2 * auc - 1

    result_dic = {
        'test_loss': test_loss.item(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc,
        'gini_index': gini_index
    }

    return probabilities[:, 1].numpy(), predicted_labels.numpy(), result_dic