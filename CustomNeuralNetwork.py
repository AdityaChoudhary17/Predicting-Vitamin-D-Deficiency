import numpy as np

class CustomNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases randomly
        self.weights_hidden = np.random.randn(input_size, hidden_size) * 0.01 # Small initialization
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid_derivative(self, a): # Derivative of sigmoid for backprop
        return a * (1 - a)

    def relu_derivative(self, a): # Derivative of ReLU for backprop
        return (a > 0).astype(float)

    def forward(self, X):
        # Hidden layer
        self.hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input) # ReLU activation

        # Output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        self.output_layer_output = self.sigmoid(self.output_layer_input) # Sigmoid for probability

        return self.output_layer_output

    def backward(self, X, y, output, learning_rate):
        # Output layer error
        output_error = output - y # Difference between prediction and true value
        output_delta = output_error * self.sigmoid_derivative(output) # Delta for output layer

        # Hidden layer error
        hidden_layer_error = np.dot(output_delta, self.weights_output.T)
        hidden_layer_delta = hidden_layer_error * self.relu_derivative(self.hidden_layer_output) # Delta for hidden layer

        # Update weights and biases (Gradient Descent)
        self.weights_output -= learning_rate * np.dot(self.hidden_layer_output.T, output_delta)
        self.bias_output -= learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_hidden -= learning_rate * np.dot(X.T, hidden_layer_delta)
        self.bias_hidden -= learning_rate * np.sum(hidden_layer_delta, axis=0, keepdims=True)

    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X_train)

            # Backward pass and update weights
            self.backward(X_train, y_train, output, learning_rate)

            # Calculate loss (Binary Cross-Entropy) - Optional for training loop, good for monitoring
            loss = self.binary_cross_entropy_loss(output, y_train)
            if epoch % 100 == 0: # Print loss every 100 epochs
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        probabilities = self.forward(X)
        # Classify based on probability threshold (0.5 for binary classification)
        predictions = (probabilities > 0.5).astype(int)
        return predictions

    def binary_cross_entropy_loss(self, y_predicted, y_true):
        # Binary Cross-Entropy Loss function
        m = len(y_true) # Number of samples
        loss = -1/m * np.sum(y_true * np.log(y_predicted) + (1 - y_true) * np.log(1 - y_predicted + 1e-8)) # Added small epsilon to avoid log(0)
        return loss

# ---  Integration with your Streamlit code ---