from typing import Any, Callable
import numpy as np


class Tensor:
    def __init__(self, data: Any, requires_grad=False, _parents=(), _op=''):
        self._data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.requires_grad = requires_grad
        self.grad = np.zeros(self._data.shape) if requires_grad else None
        self._backward: Callable[[], None] | None = None
        self._parents: tuple[Tensor, ...] = _parents
        self._op = _op

    @property
    def data(self):
        return self._data

    def __repr__(self):
        return f"Tensor(data={self._data}, requires_grad={self.requires_grad}, op='{self._op}')"

    # elementwise
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data + other._data, requires_grad=self.requires_grad or other.requires_grad,
                     _parents=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data * other._data, requires_grad=self.requires_grad or other.requires_grad,
                     _parents=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad += other._data * out.grad
            if other.requires_grad:
                other.grad += self._data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data - other._data, requires_grad=self.requires_grad or other.requires_grad,
                     _parents=(self, other), _op='-')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self._data, requires_grad=self.requires_grad,
                     _parents=(self,), _op='neg')

        def _backward():
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data / other._data, requires_grad=self.requires_grad or other.requires_grad,
                     _parents=(self, other), _op='/')

        def _backward():
            if self.requires_grad:
                self.grad += (1 / other._data) * out.grad
            if other.requires_grad:
                other.grad -= (self._data / (other._data ** 2)) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, p):
        out = Tensor(self._data ** p, requires_grad=self.requires_grad,
                     _parents=(self,), _op=f'**{p}')

        def _backward():
            if self.requires_grad:
                self.grad += (p * (self._data ** (p - 1))) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self._data), requires_grad=self.requires_grad,
                     _parents=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                self.grad += out._data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self._data), requires_grad=self.requires_grad,
                     _parents=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self._data) * out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self._data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad,
                     _parents=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad * np.ones_like(self._data)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        denom = self._data.size if axis is None else self._data.shape[axis]
        out = Tensor(self._data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad,
                     _parents=(self,), _op='mean')

        def _backward():
            if self.requires_grad:
                self.grad += (out.grad / denom) * np.ones_like(self._data)

        out._backward = _backward
        return out

    @property
    def T(self):
        out = Tensor(self._data.T, requires_grad=self.requires_grad,
                     _parents=(self,), _op='T')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T

        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data @ other._data, requires_grad=self.requires_grad or other.requires_grad,
                     _parents=(self, other), _op='@')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other._data.T
            if other.requires_grad:
                other.grad += self._data.T @ out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def relu(self):
        out = Tensor(np.maximum(0, self._data), requires_grad=self.requires_grad,
                     _parents=(self,), _op='ReLU')

        def _backward():
            if self.requires_grad:
                self.grad += (out._data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self._data)

        for node in reversed(topo):
            if node._backward:
                node._backward()

    def size(self, dim=None):
        return self._data.shape if dim is None else self._data.shape[dim]


