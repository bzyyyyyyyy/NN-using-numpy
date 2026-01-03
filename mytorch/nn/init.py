
from mytorch import Tensor


def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu') -> None:
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    gain = _calculate_gain(nonlinearity, a)
    std = gain / (fan ** 0.5)
    bound = (3.0 ** 0.5) * std
    uniform_(tensor, -bound, bound)

