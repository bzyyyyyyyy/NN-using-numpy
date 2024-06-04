import numpy as np
import neuro as nr


class Data:
    def __init__(self, file: str):
        data = np.loadtxt(file, delimiter=',', skiprows=1)
        self.labels = data[:, 0]
        self.features = data[:, 1:]
        self.len = data.shape[0]


def main():
    train = Data('archive\\fashion-mnist_train.csv')
    test = Data('archive\\fashion-mnist_test.csv')

    model = nr.sequential(nr.Linear(28*28, 128),
                  nr.ReLU(),
                  nr.Linear(128, 64),
                  nr.ReLU(),
                  nr.Linear(64, 10),
                  nr.Sigmoid())
    model.loss_fn = nr.MSE()

    batch_size = 1
    epoch = 50
    lr = .001

    for i in range(epoch):
        print(f'------- epoch {i + 1} -------')
        total_acc = 0
        total_loss = 0
        count = 0

        for n in range(train.len):
            output, loss = model.train(train.features[n], nr.to_one_hot(train.labels[n], 10))
            if (n + 1) % batch_size == 0:
                model.sgd(lr)
                model.zero_grad()

            acc = 1 if nr.from_one_hot(output) == train.labels[n] else 0

            total_loss += loss
            total_acc += acc
            count += 1

            if (n + 1) % 10000 == 0:
                print(f'train：{n + 1}, Loss: {total_loss / count}, Acc: {total_acc / count}')
                total_acc = 0
                total_loss = 0
                count = 0
                print(output)

        total_acc = 0
        total_loss = 0
        for n in range(test.len):
            output, loss = model.test(test.features[n], nr.to_one_hot(test.labels[n], 10))

            acc = 1 if nr.from_one_hot(output) == test.labels[n] else 0

            total_loss += loss
            total_acc += acc

        print(f'total test loss: {total_loss / test.len}')
        print(f'total test acc: {total_acc / test.len}')






if __name__ == '__main__':
    main()
