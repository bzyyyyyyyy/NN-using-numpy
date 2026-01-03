from collections import OrderedDict
from typing import Iterator, Tuple, Dict, Any, Optional
from ..parameter import Parameter


class Module:

    training: bool
    _parameters: Dict[str, Optional[Parameter]]
    _modules: Dict[str, Optional['Module']]

    def __init__(self):
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_modules', OrderedDict())

    def forward(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        # 可加 pre/post hook
        return self.forward(*inputs)

    def __setattr__(self, name: str, value):
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        raise NotImplementedError

        if isinstance(value, Parameter):
            remove_from(self.__dict__, self._modules)
            self._parameters[name] = value
        elif isinstance(value, Module):
            remove_from(self.__dict__, self._parameters)
            self._modules[name] = value
        else:
            remove_from(self._parameters, self._modules)
            super().__setattr__(name, value)

    def register_parameter(self, name: str, param: Optional[Parameter]):

        if '.' in name:
            raise KeyError("Parameter name can't contain '.'")
        elif name == '':
            raise KeyError("Parameter name can't be empty")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"Parameter '{name}' already registered")

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"param must be a Parameter, but got {type(param)}")
        else:
            self._parameters[name] = param

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, prefix: str = '') -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            if p is not None:
                yield prefix + name, p
        for module_name, module in self._modules.items():
            if module is not None:
                submodule_prefix = prefix + module_name + '.'
                yield from module.named_parameters(submodule_prefix)

    def register_module(self, name: str, module: Optional['Module']):
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"module must be a Module, but got {type(module)}")
        elif '.' in name:
            raise KeyError("Module name can't contain '.'")
        elif name == '':
            raise KeyError("Module name can't be empty")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"Module '{name}' already registered")

        self._modules[name] = module

    def modules(self):
        for _, m in self.named_modules():
            yield m

    def named_modules(self, prefix: str = '') -> Iterator[Tuple[str, 'Module']]:
        yield prefix, self
        for module_name, module in self._modules.items():
            if module is not None:
                submodule_prefix = prefix + module_name + '.'
                yield from module.named_modules(submodule_prefix)

    def children(self):
        for _, m in self.named_children():
            yield m

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        for name, m in self._modules.items():
            if m is not None:
                yield name, m

    def train(self, mode: bool = True):
        self.training = mode
        for m in self.children():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for _, p in self.named_parameters():
            if p.grad is not None:
                p.grad = None

    def state_dict(self) -> Dict[str, Any]:
        sd = {}
        for name, p in self.named_parameters():
            sd[name] = p.data.copy()  # numpy拷贝或相应后端
        return sd

    def load_state_dict(self, state: Dict[str, Any]):
        for name, p in self.named_parameters():
            if name not in state:
                raise KeyError(f"Missing param: {name}")
            p.data[...] = state[name]  # 就地赋值