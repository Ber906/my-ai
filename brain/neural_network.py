
import numpy as np
import json
import os

class NeuralNetwork:
    """Sariling utak - feedforward neural network with backprop"""
    
    def __init__(self, layers=[128, 64, 32, 16], learning_rate=0.01):
        self.layers = layers
        self.lr = learning_rate
        self.weights = []
        self.biases = []
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights randomly"""
        for i in range(len(self.layers) - 1):
            w = np.random.randn(self.layers[i], self.layers[i+1]) * 0.01
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward(self, X):
        """Forward pass"""
        self.activations = [X]
        self.z_values = []
        
        current = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current, w) + b
            self.z_values.append(z)
            
            if i == len(self.weights) - 1:
                current = self.sigmoid(z)  # Output layer
            else:
                current = self.relu(z)     # Hidden layers
            
            self.activations.append(current)
        
        return current
    
    def backward(self, X, y, output):
        """Backpropagation - learning happens here"""
        m = X.shape[0]
        deltas = []
        
        # Output layer error
        error = output - y
        delta = error * self.sigmoid_derivative(output)
        deltas.append(delta)
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(deltas[-1], self.weights[i+1].T)
            delta = error * self.relu_derivative(self.activations[i+1])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * np.dot(self.activations[i].T, deltas[i]) / m
            self.biases[i] -= self.lr * np.mean(deltas[i], axis=0, keepdims=True)
    
    def train(self, X, y, epochs=100):
        """Train the network"""
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 20 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """Make prediction"""
        return self.forward(X)
    
    def save(self, path="brain_weights.json"):
        """Save learned weights"""
        data = {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'layers': self.layers
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path="brain_weights.json"):
        """Load learned weights"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.weights = [np.array(w) for w in data['weights']]
            self.biases = [np.array(b) for b in data['biases']]
            return True
        return False
