import numpy as np
import mytorch as t


def test_tensor_autograd():
    # Test addition
    a = t.tensor([2.0, 3.0], requires_grad=True)
    b = t.tensor([4.0, 5.0], requires_grad=True)
    c = a + b
    c.sum().backward()
    assert np.allclose(a.grad, [1.0, 1.0]), f"Addition grad failed for a: {a.grad}"
    assert np.allclose(b.grad, [1.0, 1.0]), f"Addition grad failed for b: {b.grad}"

    # Test multiplication
    a = t.tensor([2.0, 3.0], requires_grad=True)
    b = t.tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    c.sum().backward()
    assert np.allclose(a.grad, [4.0, 5.0]), f"Multiplication grad failed for a: {a.grad}"
    assert np.allclose(b.grad, [2.0, 3.0]), f"Multiplication grad failed for b: {b.grad}"

    # Test power
    a = t.tensor([2.0, 3.0], requires_grad=True)
    c = a ** 2
    c.sum().backward()
    assert np.allclose(a.grad, [4.0, 6.0]), f"Power grad failed for a: {a.grad}"

    # Test matrix multiplication
    a = t.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = t.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = a @ b
    c.sum().backward()
    assert np.allclose(a.grad, [[11.0, 15.0], [11.0, 15.0]]), f"Matmul grad failed for a: {a.grad}"
    assert np.allclose(b.grad, [[4.0, 4.0], [6.0, 6.0]]), f"Matmul grad failed for b: {b.grad}"

    print("All autograd tests passed!")


if __name__ == "__main__":
    test_tensor_autograd()
