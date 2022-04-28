import torch
from torch import nn
from torchinfo import summary


class CRNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / lstm / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.BatchNorm2d(128),
            nn.Dropout(p=0.2)
        )
        self.lstm1 = nn.LSTM(10, 32, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(32, 32, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data, h_n, h_c): # , h_n, h_c
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.reshape(x, (-1, 128, 10))
        x, (h_n, h_c) = self.lstm1(x, None) # (h_n, h_c)
        # x, (h_n, h_c) = self.lstm2(x, None)  # (h_n, h_c)
        x = self.dropout(x)
        # x = x[:, -1, :]

        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)
        return x, h_n, h_c

if __name__ == "__main__":
    crnn = CRNNNetwork()
    summary(crnn.cuda(), (32, 1, 180, 18))
    # print(cnn)


