#import numpy, matplot, pandas, and sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd  

# Load the datasets
file_paths = ['groupA.txt', 'groupB.txt', 'groupC.txt']
def load_dataset(file_path):
    data = pd.read_csv(file_path, header=None)
    data = data.values  # Convert DataFrame to NumPy array
    
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, y

# Perceptron class definition
class Perceptron:
    def __init__(self, N, alpha=0.1):
        self.W = np.random.uniform(-0.5, 0.5, size=N + 1)  # +1 for bias term
        self.alpha = alpha
# Perceptron Step function
    def step(self, x):
        return np.where(x >= 0, 1, 0)
# Perceptron Sigmoid function
    def sigmoid(self, x, gain=1):
        return 1 / (1 + np.exp(-gain * x))
# Perceptron Predict function
    def predict(self, X, activation='hard', gain=1):
        X = np.insert(X, 0, 1, axis=1)
        net_input = np.dot(X, self.W)
        if activation == 'hard':
            return self.step(net_input)
        elif activation == 'soft':
            return self.sigmoid(net_input, gain=gain)
        else:
            raise ValueError("Activation function must be 'hard' or 'soft'")
# Perceptron Fit function
    def fit(self, X, y, epochs=5000, epsilon=1e-5, activation='hard', gain=1):
        X = np.insert(X, 0, 1, axis=1)
        total_errors = []
        for epoch in range(epochs):
            total_error = 0
            for xi, target in zip(X, y):
                net_input = np.dot(xi, self.W)
                if activation == 'hard':
                    output = self.step(net_input)
                    error = target - output
                    self.W += self.alpha * error * xi
                elif activation == 'soft':
                    output = self.sigmoid(net_input, gain=gain)
                    error = target - output
                    self.W += self.alpha * error * xi * output * (1 - output)
                total_error += error ** 2
            total_errors.append(total_error)
            if total_error < epsilon:
                print(f'Converged at epoch {epoch}')
                break
        return total_errors

# Plot results 
def plot_decision_boundary(X, y, perceptron, activation='hard', gain=1, title=''):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = perceptron.predict(grid, activation=activation, gain=gain)
    if activation == 'soft':
        Z = (Z >= 0.5).astype(int) 
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, levels=[-0.1, 0.5, 1.1], colors=['#FFAAAA', '#AAAAFF'])
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.show()

# Main execution
for file_path in file_paths:
    # Load and normalize the dataset
    X, y = load_dataset(file_path)
    
    y = (y > 0).astype(int)
    
    # Step 1: 75% training, 25% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
    N = X_train.shape[1]
    
    # Initialize perceptron with hard activation function
    perceptron_hard = Perceptron(N, alpha=0.01)
    
    # Train perceptron
    total_errors = perceptron_hard.fit(X_train, y_train, epochs=5000, epsilon=1e-5, activation='hard')
    print(f"Final Total Error (TE) for training (hard activation): {total_errors[-1]}")
    
    # Predict on test data
    y_pred = perceptron_hard.predict(X_test, activation='hard')
    
    # Plot training data and decision boundary
    plot_decision_boundary(X_train, y_train, perceptron_hard, activation='hard', title=f'Training Data and Decision Boundary ({file_path})')
    
    # Plot testing data and decision boundary
    plot_decision_boundary(X_test, y_test, perceptron_hard, activation='hard', title=f'Testing Data and Decision Boundary ({file_path})')
    
    # Create confusion matrix and classification report
    print(f"Results for {file_path} with 75% training data (hard activation):")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 2: 25% training, 75% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25)
    N = X_train.shape[1]
    
    # Initialize perceptron with hard activation function
    perceptron_hard = Perceptron(N, alpha=0.01)
    
    # Train perceptron
    total_errors = perceptron_hard.fit(X_train, y_train, epochs=5000, epsilon=1e-5, activation='hard')
    print(f"Final Total Error (TE) for training (hard activation): {total_errors[-1]}")
    
    # Predict on test data
    y_pred = perceptron_hard.predict(X_test, activation='hard')
    
    # Plot training data and decision boundary
    plot_decision_boundary(X_train, y_train, perceptron_hard, activation='hard', title=f'Training Data and Decision Boundary ({file_path})')
    
    # Plot testing data and decision boundary
    plot_decision_boundary(X_test, y_test, perceptron_hard, activation='hard', title=f'Testing Data and Decision Boundary ({file_path})')
    
    # Create confusion matrix and classification report
    print(f"Results for {file_path} with 25% training data (hard activation):")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Repeat the above steps for soft unipolar activation function
    # Adjust alpha and gain as necessary
    # Initialize perceptron with soft activation function
    perceptron_soft = Perceptron(N, alpha=0.01)
    
    # Train perceptron
    total_errors = perceptron_soft.fit(X_train, y_train, epochs=5000, epsilon=1e-5, activation='soft', gain=1)
    print(f"Final Total Error (TE) for training (soft activation): {total_errors[-1]}")
    
    # Predict on test data
    y_pred = perceptron_soft.predict(X_test, activation='soft', gain=1)
    y_pred_labels = (y_pred >= 0.5).astype(int)  # Convert probabilities to class labels
    
    # Plot training data and decision boundary
    plot_decision_boundary(X_train, y_train, perceptron_soft, activation='soft', gain=1, title=f'Training Data and Decision Boundary ({file_path})')
    
    # Plot testing data and decision boundary
    plot_decision_boundary(X_test, y_test, perceptron_soft, activation='soft', gain=1, title=f'Testing Data and Decision Boundary ({file_path})')
    
    # Create confusion matrix and classification report
    print(f"Results for {file_path} with 25% training data (soft activation):")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_labels))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_labels))
