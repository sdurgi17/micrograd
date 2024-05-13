import math
import random
from typing import Any


class Value:
    def __init__(self, val, child_nodes=set(), op=None) -> None:
        self.val = val
        self._child_nodes = child_nodes
        self._op = op
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value({self.val})"

    def __add__(self, other) -> "Value":
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.val + other.val, [self, other], "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Value":
        other = Value(other) if not isinstance(other, Value) else other
        out = Value(self.val * other.val, [self, other], "*")

        def _backward():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.val**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.val ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other) -> "Value":
        return self.__mul__(other)

    def __truediv__(self, other):  # self / other
        return self * other**-1

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __radd__(self, other) -> "Value":
        return self + other

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad  # NOTE: in the video I incorrectly used = instead of +=. Fixed here.

        out._backward = _backward

        return out

    def tanh(self):
        x = self.val
        t = math.tanh(x)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        self.grad = 1.0
        topo_sorted = []
        visited = set()

        def topo(node, results):
            if node not in visited:
                visited.add(node)
                for child in node._child_nodes:
                    topo(child, results)
                results.append(node)

        topo(self, topo_sorted)
        for node in reversed(topo_sorted):
            node._backward()
            print(f"grad : {node.grad}, val: {node.val}, op: {node._op}")


def relu(x):
    return max(0, x)


class Neuron:
    def __init__(self, nin, activation_fn=None) -> None:
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x) -> Any:
        act = sum([w * x for w, x in zip(self.weights, x)], self.bias)
        return act.tanh()

    def parameters(self):
        return self.weights + [self.bias]


class Layer:
    def __init__(self, nin, nout, **kwargs) -> None:
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x) -> Any:
        return [neuron(x) for neuron in self.neurons]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]


class MLP:
    def __init__(self, nin, nouts, **kwargs) -> None:
        layer_sizes = [nin] + nouts
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1], **kwargs) for i in range(len(layer_sizes) - 1)]

    def __call__(self, x) -> Any:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]


if __name__ == "__main__":

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

    random.seed(123)
    mlp = MLP(3, [3, 4, 4, 1])
    print(mlp(xs[0]))
    print(mlp.parameters())
    losses = []
    for i in range(1000):
        yp = [mlp(x)[0] for x in xs]
        loss = sum([(yh - y) ** 2 for y, yh in zip(ys, yp)])
        losses.append(loss.val)
        for p in mlp.parameters():
            p.grad = 0.0

        loss.backward()
        print(i, loss.val)

        for p in mlp.parameters():
            p.val += -0.1 * p.grad

    print(losses)
    import matplotlib.pyplot as plt

    plt.plot(losses)
    yp = [mlp(x)[0].val for x in xs]
    print(yp)
