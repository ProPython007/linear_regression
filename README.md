# linear_regression
Performing linear regression on the dataset using numpy only (no sklearn).

import numpy as np

# Load the data
data = np.loadtxt("data.csv", delimiter=",")

# Split the data into features and labels
features = data[:, :-1]
labels = data[:, -1]

# Initialize the model parameters
theta = np.zeros(features.shape[1])

# Define the sigmoid function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Define the cost function
def cost_function(theta, features, labels):
  h = sigmoid(features @ theta)
  J = -1 / features.shape[0] * np.sum(labels * np.log(h) + (1 - labels) * np.log(1 - h))
  return J

# Define the gradient descent algorithm
def gradient_descent(theta, features, labels, learning_rate, iterations):
  costs = []
  for i in range(iterations):
    h = sigmoid(features @ theta)
    gradient = (features.T @ (h - labels)) / features.shape[0]
    theta -= learning_rate * gradient
    costs.append(cost_function(theta, features, labels))
  return theta, costs

# Train the model
theta, costs = gradient_descent(theta, features, labels, 0.01, 1000)

# Plot the cost function
plt.plot(costs)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()

# Make predictions
predictions = sigmoid(features @ theta) > 0.5

# Evaluate the model
accuracy = np.mean(predictions == labels)
print("Accuracy:", accuracy)
