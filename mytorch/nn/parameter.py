from mytorch import Tensor


class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad)
        self.is_parameter = True
