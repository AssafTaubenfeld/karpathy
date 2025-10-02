import math


class Value:
    def __init__(self, data, _childern=(), _op='', label = ''):
        """
        Initialize a Value node in the computation graph.
        
        Args:
            data (float): The scalar value stored in this node
            _childern (tuple): Tuple of Value objects that are inputs to this node
            _op (str): String representing the operation that produced this node (e.g., '+', '*', 'tanh')
            label (str): Optional label for visualization and debugging
        
        Attributes:
            grad (float): Gradient of the output with respect to this node, initialized to 0
            _backward (function): Function that computes and accumulates gradients for the
                                 children nodes. Initially set to no-op, but overridden by
                                 operations to implement their specific backward pass logic.
            _prev (set): Set of Value nodes that are direct inputs to this node
        """
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_childern)
        self._op = _op
        self.label = label
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            # Compute the gradients, based on the chain rule for addition
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data * other.data, (self, other), '*')
        def _backward():
            # Compute the gradients, based on the chain rule for multiplication
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other): # self /  other
        return self * other**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other -1)) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n)-1) / (math.exp(2*n) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            # Compute the gradients, based on the chain rule and tanh diverative
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        """
        Compute gradients for all nodes via backpropagation.
        
        Performs reverse-mode automatic differentiation by:
        1. Building topological order of all nodes in the graph
        2. Setting this node's gradient to 1.0
        3. Propagating gradients backward through each node's _backward() function
        
        After execution, each Value's .grad contains the derivative of this node w.r.t. that Value.
        """
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
            topo.append(v)
        build_topo(self)

        self.grad = 1
        for node in reversed(topo):
            node._backward()