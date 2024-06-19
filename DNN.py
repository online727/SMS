import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义 Maxout 单元
class Maxout(nn.Module):
    def __init__(self, in_features, out_features, num_pieces):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features, out_features) for _ in range(num_pieces)
        ])

    def forward(self, x):
        outputs = [self.linear_layers[i](x) for i in range(self.num_pieces)]
        max_output = torch.max(torch.stack(outputs), 0)[0]
        return max_output


# Network
class MaxoutNetWithRegularization(nn.Module):
    def __init__(self, dropout_input=0.1, dropout_hidden=0.5, l1_lambda=0.00001):
        super(MaxoutNetWithRegularization, self).__init__()
        self.l1_lambda = l1_lambda
        self.maxout1 = Maxout(31, 31, 2) 
        self.dropout1 = nn.Dropout(p=dropout_input)  # 输入层 dropout
        self.maxout2 = Maxout(31, 10, 2)
        self.dropout2 = nn.Dropout(p=dropout_hidden)  # 隐藏层 dropout
        self.maxout3 = Maxout(10, 5, 2)
        self.dropout3 = nn.Dropout(p=dropout_hidden)  # 隐藏层 dropout
        self.sigmoid_output = nn.Linear(5, 2)

    def forward(self, x):
        x = self.dropout1(x)
        x = self.maxout1(x)
        x = self.dropout2(x)
        x = self.maxout2(x)
        x = self.dropout3(x)
        x = self.maxout3(x)
        x = self.sigmoid_output(x)
        return x
    
    # 计算 L1 正则化损失
    def l1_regularization(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return self.l1_lambda * l1_loss

if __name__ == "__main__":
    net_with_regularization = MaxoutNetWithRegularization()
    print(net_with_regularization)  # print the network architecture

    # example
    example_input = torch.randn(1, 31)
    example_output = net_with_regularization(example_input)
    l1_loss = net_with_regularization.l1_regularization()

    print(example_output)
    print("L1 Regularization Loss:", l1_loss.item())