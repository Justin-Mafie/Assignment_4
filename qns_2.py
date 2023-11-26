import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Scale the target variable to be in the range (0, 1) using Min-Max scaling
scaler = MinMaxScaler()
y = y.reshape(-1, 1)
y = scaler.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Define the neural network model with one hidden layer including 8 neurons
def create_model(use_relu_output=False):
    # Create a Sequential model
    model = Sequential()

    # Add a hidden layer with 8 neurons and 'relu' activation function
    model.add(Dense(8, input_dim=X_train.shape[1], activation='relu'))

    # Add an output layer with 'relu' or 'linear' activation based on the argument
    if use_relu_output:
        model.add(Dense(1, activation='relu'))
    else:
        model.add(Dense(1, activation='linear'))  # Use 'linear' for regression problems

    # Compile the model with Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# Train the model with 'relu' activation function for the output layer
model_with_relu = create_model(use_relu_output=True)
model_with_relu.fit(X_train, y_train, epochs=10, verbose=1)

# Train the model without 'relu' activation function for the output layer
model_without_relu = create_model(use_relu_output=False)
model_without_relu.fit(X_train, y_train, epochs=10, verbose=1)

# Make predictions on the testing set
y_pred_with_relu = model_with_relu.predict(X_test)
y_pred_without_relu = model_without_relu.predict(X_test)

# Invert the scaling to get predictions in the original scale
y_pred_with_relu = scaler.inverse_transform(y_pred_with_relu)
y_pred_without_relu = scaler.inverse_transform(y_pred_without_relu)
y_test_original_scale = scaler.inverse_transform(y_test)

# Measure the performance using mean absolute error
mae_with_relu = mean_absolute_error(y_test_original_scale, y_pred_with_relu)
mae_without_relu = mean_absolute_error(y_test_original_scale, y_pred_without_relu)

# Print the results
print(f'MAE with relu: {mae_with_relu}')
print(f'MAE without relu: {mae_without_relu}')
