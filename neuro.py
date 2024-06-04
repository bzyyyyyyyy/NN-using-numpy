import numpy as np


def avg_add(a, avg, n):
    return (avg * n + a) / (n + 1)


def to_one_hot(label, num_classes):
    arr = np.zeros(num_classes)
    arr[int(label)] = 1
    return arr


def from_one_hot(one_hot):
    labels = np.argmax(one_hot, axis=0)
    return labels


def he_init(shape):
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, shape)


def xavier_init(shape):
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape)


def kaiming_uniform(shape):
    limit = np.sqrt(6 / shape[0])
    return np.random.uniform(-limit, limit, size=shape)


class Model:
    def __init__(self):
        self._modules = {}
        self.loss_fn = None

    def add_module(self, name: str, module):
        self._modules[name] = module

    def __call__(self, input: np.ndarray):
        for m in self._modules.values():
            input = m(input)
        return input

    def test(self, get: np.ndarray, exp: np.ndarray) -> (np.ndarray, np.ndarray):
        for m in self._modules.values():
            get = m(get)
        loss = self.loss_fn(get, exp)
        return get, loss

    def train(self, get: np.ndarray, exp: np.ndarray) -> (np.ndarray, np.ndarray):
        for m in self._modules.values():
            get = m(get)
        loss = self.loss_fn(get, exp)
        self.backward()
        return get, loss

    def backward(self):
        out_grad = self.loss_fn.grad()
        # print(f'grad -> loss:\n{out_grad}')
        for m in reversed(self._modules.values()):
            # print('\n')
            out_grad = m.grad(out_grad)
        # print(f'grad -> m1:\n{out_grad}')

    def zero_grad(self):
        for m in reversed(self._modules.values()):
            m.zero_grad()

    def sgd(self, lr):
        for m in reversed(self._modules.values()):
            m.sgd(lr)


class Linear:
    def __init__(self, inD: int, outD: int):
        self.inD = inD
        self.outD = outD
        self.weight = kaiming_uniform([outD, inD])
        self.bias = np.zeros(outD)
        self.input: np.ndarray
        self.weight_grad = np.zeros((outD, inD))
        self.bias_grad = np.zeros(outD)
        self.grad_t = 0

    def __call__(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        # print(self.weight)
        return self.weight @ input + self.bias

    def grad(self, out_grad: np.ndarray) -> np.ndarray:
        # print(f'Linear -> grad:\n{out_grad}')
        # print(f'Linear in:\n{self.input}')
        self.weight_grad = avg_add(np.outer(out_grad, self.input), self.weight_grad, self.grad_t)
        # print(f'Linear weight_grad:\n{self.weight_grad}')
        self.bias_grad = avg_add(out_grad, self.bias_grad, self.grad_t)
        # print(f'Linear bias_grad:\n{self.bias_grad}')
        # print(f'Linear weight:\n{self.weight}')
        self.grad_t += 1
        return np.dot(out_grad, self.weight)

    def zero_grad(self):
        self.weight_grad = np.zeros((self.outD, self.inD))
        self.bias_grad = np.zeros(self.outD)

    def sgd(self, lr):
        # print(self.weight_grad)
        self.weight -= lr * self.weight_grad
        # print(self.bias_grad)
        self.bias -= lr * self.bias_grad


class ReLU:
    def __init__(self):
        self.input: np.ndarray

    def __call__(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(0, input)

    def grad(self, out_grad: np.ndarray) -> np.ndarray:
        # print(f'ReLU -> grad:\n{out_grad}')
        # print(f'ReLU in:\n{self.input}')
        return np.where(self.input > 0, 1, 0) * out_grad

    def zero_grad(self):
        pass

    def sgd(self, lr):
        pass


class Sigmoid:
    def __init__(self):
        self.output: np.ndarray

    def __call__(self, input: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def grad(self, out_grad: np.ndarray) -> np.ndarray:
        # print(f'Sig -> grad:\n{out_grad}')
        # print(f'Sig out:\n{self.output}')
        return (self.output * (1 - self.output)) * out_grad

    def zero_grad(self):
        pass

    def sgd(self, lr):
        pass


class Softmax:
    def __init__(self):
        self.output: np.ndarray

    def __call__(self, input: np.ndarray) -> np.ndarray:
        a = np.exp(input - np.max(input))
        self.output = a / np.sum(a)
        return self.output

    def grad(self, out_grad: np.ndarray) -> np.ndarray:
        s = self.output.reshape(-1, 1)
        return (np.diagflat(s) - np.dot(s, s.T)) @ out_grad

    def zero_grad(self):
        pass

    def sgd(self, lr):
        pass


class MSE:
    def __init__(self):
        self.get: np.ndarray
        self.exp: np.ndarray
        self.n: int

    def __call__(self, get: np.ndarray, exp: np.ndarray) -> np.ndarray:
        self.get = get
        self.exp = exp
        self.n = get.size
        return np.mean((get - exp) ** 2)

    def grad(self) -> np.ndarray:
        # print(f'MSE get: \n{self.get}')
        # print(f'MSE exp: \n{self.exp}')
        return 2 * (self.get - self.exp) / self.n


class sequential(Model):
    def __init__(self, *args):
        super().__init__()
        for i, m in enumerate(args):
            self.add_module(str(i), m)


def main():
    model = sequential(Linear(2, 3),
                       ReLU(),
                       Linear(3, 2),
                       Sigmoid())
    model.loss_fn = MSE()
    a = np.array([-1, 2])
    b = np.array([0, 1])
    # l = Linear(2, 2)
    # print(l(a))

    # for n in range(10):
    #     for i in range(10):
    #         l = model.train(a, b)
    #         print(l)
    #     model.sgd(.1)
    #     model.zero_grad()


if __name__ == '__main__':
    main()
