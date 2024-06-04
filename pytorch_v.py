import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as f
import matplotlib.pyplot as plt


class Data:
    def __init__(self, file: str):
        data = np.loadtxt(file, delimiter=',', skiprows=1)
        self.labels = data[:, 0]
        self.features = data[:, 1:]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.Linear(64, 32),
            # nn.ReLU(),
            nn.Linear(64, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        # x = f.softmax(x, dim=1)
        return x



def main():
    train = Data('archive\\fashion-mnist_train.csv')
    # print(torch.tensor(train.labels).type(torch.int) - 1)
    # return
    train_dataset = TensorDataset(torch.tensor(train.features).type(torch.float32), f.one_hot(torch.tensor(train.labels).type(torch.int64), num_classes=10))

    test = Data('archive\\fashion-mnist_test.csv')
    test_dataset = TensorDataset(torch.tensor(test.features).type(torch.float32), f.one_hot(torch.tensor(test.labels).type(torch.int64), num_classes=10))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # print(train_dataset[0])
    # return

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = Net().to(device)
    print(model)

    # loss_fn = nn.CrossEntropyLoss().to(device)
    loss_fn = nn.MSELoss().to(device)
    print(loss_fn)

    optimizer = torch.optim.SGD(model.parameters(), lr=.001)

    epoch = 500

    total_train_step = 0

    total_test_step = 0

    with SummaryWriter("./logs_train") as writer:
        for i in range(epoch):
            print("------- epoch {} -------".format(i + 1))

            # start training
            model.train()
            for data in train_loader:
                features, labels = data
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)
                # print(outputs)
                # print(labels)
                loss = loss_fn(outputs, labels.type(torch.float32))

                # optimize model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_step += 1

                total_train_accuracy = 0
                for n in range(len(outputs)):
                    _, a = torch.max(outputs[n], 0)
                    _, b = torch.max(labels[n], 0)
                    accuracy = 1 if a == b else 0
                    total_train_accuracy += accuracy

                if total_train_step % 10 == 0:
                    train_accuracy = total_train_accuracy / len(outputs)
                    writer.add_scalar("train_loss", loss.item(), total_train_step)
                    writer.add_scalar("train_accuracy", train_accuracy, total_train_step)
                    if total_train_step % 100 == 0:
                        print("train：{}, Loss: {}, Accuracy: {}".format(total_train_step, loss.item(), train_accuracy))

            model.eval()
            total_test_loss = 0
            total_accuracy = 0
            with torch.no_grad():
                for data in test_loader:
                    features, labels = data
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = model(features)
                    loss = loss_fn(outputs, labels.type(torch.float32))
                    total_test_loss += loss.item()
                    for n in range(len(outputs)):
                        _, a = torch.max(outputs[n], 0)
                        _, b = torch.max(labels[n], 0)
                        accuracy = 1 if a == b else 0
                        total_accuracy += accuracy

            print("total test loss: {}".format(total_test_loss))
            print("total test accuracy: {}".format(total_accuracy / len(outputs)))
            writer.add_scalar("test_loss", total_test_loss, total_test_step)
            writer.add_scalar("test_accuracy", total_accuracy / len(outputs), total_test_step)
            total_test_step += 1

if __name__ == '__main__':
    main()
