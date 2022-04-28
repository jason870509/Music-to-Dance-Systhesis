import torch
import torchaudio
import numpy as np
import torch.utils.data as Data
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crnn import CRNNNetwork
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)

if __name__ == '__main__':
    print(f"Using {device}")
    melgrams = np.load('data/mfcc/melgrams.npy')
    labels = np.load('data/mfcc/labels.npy')
    X_train, X_test, y_train, y_test = train_test_split(melgrams, labels, test_size=0.2)
    print(X_train.shape, y_train.shape)
    X_test_length = X_test.shape[0]

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

    optimizer = torch.optim.Adam(crnn.parameters(), lr=LEARNING_RATE)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss().to(device)  # the target label is not one-hotted
    h_n, h_c = torch.randn(2, 32, 32).to(device), torch.randn(2, 32, 32).to(device)

    # training and testing
    loss = 0
    best_loss = 100
    writer = SummaryWriter('logs')

    for epoch in range(EPOCHS):
        crnn.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):  # gives batch data, normalize x when iterate train_loade
            b_x, b_y = b_x.to(device), b_y.to(device, dtype=torch.long)
            b_x = torch.reshape(b_x, (-1, 1, 180, 18))

            output, h_n, h_c = crnn(b_x, h_n, h_c)
            h_n, h_c = h_n.data, h_c.data

            loss = loss_func(output, b_y)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        crnn.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for b_x, b_y in test_dataloader:
                b_x, b_y = b_x.to(device), b_y.to(device)
                b_x = torch.reshape(b_x, (-1, 1, 180, 18))
                b_y = b_y.to(device, dtype=torch.long)
                test_output, _, _ = crnn(b_x, h_n, h_c)

                # 計算loss
                loss = loss_func(test_output, b_y)
                total_test_loss += loss
                # 計算accuracy
                accuracy = (test_output.argmax(1) == b_y).sum()
                total_accuracy += accuracy

        test_loss = total_test_loss / (X_test_length//BATCH_SIZE)
        if best_loss > test_loss:
            best_loss = test_loss
            torch.save(crnn.state_dict(), "model/crnn.pth")
            print("Save model with loss {}.".format(best_loss))

        if epoch % 10 == 0:
            print('Epoch: ', epoch)
            print("Train loss: {}".format(loss))
            print("Total loss: {}".format(total_test_loss / (X_test_length//BATCH_SIZE)))
            print("Total accuracy: {}".format(total_accuracy / X_test_length))
            writer.add_scalars('train test loss', {'train_loss': loss.item(), 'test_loss': test_loss}, epoch)
            writer.add_scalar('train accuracy', total_accuracy / X_test_length, epoch)

    writer.close()
