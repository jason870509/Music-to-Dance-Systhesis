import torch
import torchaudio
import numpy as np
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crnn import CRNNNetwork
from sklearn.model_selection import train_test_split

BATCH_SIZE = 10
EPOCHS = 300
LEARNING_RATE = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

if __name__ == '__main__':
    print(f"Using {device}")
    melgrams = np.load('norm_mfcc_mint/melgrams.npy')
    labels = np.load('norm_mfcc_mint/labels.npy')
    X_train, X_test, y_train, y_test = train_test_split(melgrams, labels, test_size=0.2)
    X_test_length = X_test.shape[0]
    print(X_test_length)
    # numpy --> tensor
    tensor_X_train, tensor_y_train = torch.Tensor(X_train), torch.Tensor(y_train)
    tensor_X_test, tensor_y_test = torch.Tensor(X_test), torch.Tensor(y_test)
    # tensor --> tensor dataset
    train_dataset = Data.TensorDataset(tensor_X_train, tensor_y_train)
    test_dataset = Data.TensorDataset(tensor_X_test, tensor_y_test)
    # tensor dataset --> dataloader
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=0, drop_last=True)

    crnn = CRNNNetwork().to(device)
    print(crnn)  # net architecture
    weight = torch.load('model/crnn.pth')
    crnn.load_state_dict(weight)
    h_n, h_c = torch.randn(2, 32, 32).to(device), torch.randn(2, 32, 32).to(device)

    # print 10 predictions from test data
    crnn.eval()
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x, b_y = b_x.to(device), b_y.to(device, dtype=torch.long)
            b_x = torch.reshape(b_x, (-1, 1, 180, 18))

            test_output, _, _ = crnn(b_x[:10], h_n, h_c)
            pred_y = test_output.argmax(1)
            print(pred_y, 'prediction number')
            print(b_y[:10], 'real number')

