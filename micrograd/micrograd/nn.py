import random
from micrograd.engine import Value

class Module:
    """
    Base class for all neural network modules.
    
    Provides common functionality for managing parameters and gradients.
    All neural network components (Neuron, Layer, MLP) inherit from this class.
    """

    def zero_grad(self):
        """Reset gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0
            
    def parameters(self):
        """Return list of all trainable parameters. Override in subclasses."""
        return []

class Neuron(Module):
    """
    Single neuron that computes tanh(w·x + b).
    
    Attributes:
        w: List of weight Values for each input
        b: Bias Value
    """

    def __init__(self, num_inputs_for_neuron):
        """
        Initialize neuron with random weights and bias.
        
        Args:
            num_inputs_for_neuron (int): Number of inputs
        """
        self.w = [Value(random.uniform(-1,1)) for _ in range(num_inputs_for_neuron)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        """
        Forward pass: computes weighted sum + bias, applies tanh.
        
        Args:
            x (list): Input Value objects
            
        Returns:
            Value: Activated output
        """
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b) # w * x + b
        out = act.tanh()  # apply activation
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    """
    Layer of neurons that process the same inputs in parallel.
    
    Attributes:
        neurons (list): Neuron objects in this layer
    """

    def __init__(self, num_inputs_for_neuron, num_nueurons_in_layer):
        """
        Initialize layer with multiple neurons.
        
        Args:
            num_inputs_for_neuron (int): Inputs per neuron
            num_nueurons_in_layer (int): Number of neurons
        """
        self.neurons = [Neuron(num_inputs_for_neuron) for _ in range (num_nueurons_in_layer)]
    
    def __call__(self, x):
        """
        Forward pass through all neurons.
        
        Args:
            x (list): Input Value objects (from prev layer)
            
        Returns:
            list: Output Value from each neuron
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        
    def parameters(self):
        params = []
        for neuron in self.neurons:
          p = neuron.parameters()
          params.extend(p)
        return params

    
class MLP(Module):
    """
    Multi-Layer Perceptron: sequential feedforward neural network.
    
    Attributes:
        layers (list): Layer objects that make up the network
    """
    
    def __init__(self, num_inputs, num_outputs_array):
        """
        Initialize MLP with specified architecture.
    
        Args:
            num_inputs (int): Number of input features
            num_outputs_array (list): Neurons per layer, e.g. [16, 16, 1]
        """
        # Build list of all layer sizes: [input_size, hidden1, hidden2, ..., output]
        layer_sizes = [num_inputs] + num_outputs_array
    
        # Create layers by pairing consecutive sizes: (3,4), (4,4), (4,1)
        self.layers = [Layer(n_in, n_out) 
                    for n_in, n_out in zip(layer_sizes, layer_sizes[1:])]
    
    def __call__(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (list): Input Value objects
            
        Returns:
            list or Value: Network output
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]