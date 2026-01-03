from mytorch import Tensor
from typing import Any


def tensor(data: Any, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)
