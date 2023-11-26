import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load the pre-trained neural network model from the saved file ('iris_model.h5')
model = load_model('iris_model.h5')  # Make sure 'iris_model.h5' is in the correct path

# Evaluate the model on the test set
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# Convert true labels to one-hot encoding for comparison
true_labels_one_hot = np.zeros((y_test.size, y_test.max()+1))
true_labels_one_hot[np.arange(y_test.size), y_test] = 1

# Calculate accuracy
accuracy = accuracy_score(true_labels_one_hot.argmax(axis=1), predicted_labels)

# Print the accuracy on the test set
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
