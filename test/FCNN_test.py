import numpy as np

compare_torch = True

if compare_torch:
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter
    import torch.nn.functional as f
else:
    import mytorch as torch
    from mytorch.utils.data import TensorDataset, DataLoader



class Data:
    def __init__(self, file: str):
        data = np.loadtxt(file, delimiter=',', skiprows=1)
        self.labels = data[:, 0]
        self.features = data[:, 1:]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x



def main():
    train = Data('..\\archive\\fashion-mnist_train.csv')
    # print(torch.tensor(train.labels).type(torch.int) - 1)
    # return
    train_dataset = TensorDataset(torch.tensor(train.features).type(torch.float32), f.one_hot(torch.tensor(train.labels).type(torch.int64), num_classes=10))

    test = Data('..\\archive\\fashion-mnist_test.csv')
    test_dataset = TensorDataset(torch.tensor(test.features).type(torch.float32), f.one_hot(torch.tensor(test.labels).type(torch.int64), num_classes=10))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # print(train_dataset[0])
    # return


    model = Net()

    print(model)

    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()
    print(loss_fn)

    optimizer = torch.optim.SGD(model.parameters(), lr=.005)

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
                outputs = model(features)
                # print(outputs)
                # print(labels)
                loss = loss_fn(outputs, labels.type(torch.float32))

                # optimize model

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

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


            # start testing
            model.eval()
            total_test_loss = 0
            total_accuracy = 0
            batches = 0
            with torch.no_grad():
                for data in test_loader:
                    batches += 1
                    features, labels = data
                    outputs = model(features)
                    loss = loss_fn(outputs, labels.type(torch.float32))
                    total_test_loss += loss.item()
                    for n in range(len(outputs)):
                        _, a = torch.max(outputs[n], 0)
                        _, b = torch.max(labels[n], 0)
                        accuracy = 1 if a == b else 0
                        total_accuracy += accuracy

            print("total test loss: {}".format(total_test_loss))
            print("total test accuracy: {}".format(total_accuracy / len(outputs) / batches))
            writer.add_scalar("test_loss", total_test_loss, total_test_step)
            writer.add_scalar("test_accuracy", total_accuracy / len(outputs) / batches, total_test_step)
            total_test_step += 1


if __name__ == '__main__':
    main()
