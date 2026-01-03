from mytorch import Tensor


def relu(input: Tensor) -> Tensor:
    return input.relu()


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    out = input @ weight.T
    if bias is not None:
        out = out + bias
    return out


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return ((input - target) ** 2).mean()


def logsigmoid(input: Tensor) -> Tensor:
    return 1 / (1 + (-input).exp())


def softplus(input: Tensor) -> Tensor:
    return (1 + input.exp()).log()


def softmax(input: Tensor, dim: int) -> Tensor:
    exp_input = input.exp()
    sum_exp = exp_input.sum(axis=dim, keepdims=True)
    return exp_input / sum_exp


# def cross_entropy_loss(input: Tensor, target: Tensor) -> Tensor:
#     log_softmax = (input - softmax(input, dim=1).log())
#     loss = - (target * log_softmax).sum(axis=1).mean()
#     return loss

