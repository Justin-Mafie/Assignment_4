import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Get the number of instances in the training and testing sets
number_of_training_instances = X_train.shape[0]
number_of_testing_instances = X_test.shape[0]

# Reshape the dataset to flatten the images
X_train = X_train.reshape((number_of_training_instances, 28*28))
X_test = X_test.reshape((number_of_testing_instances, 28*28))

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define the neural network model
model = Sequential()
model.add(Dense(128, input_dim=28*28, activation='relu'))  # First hidden layer with 128 neurons and 'relu' activation
model.add(Dense(64, activation='relu'))  # Second hidden layer with 64 neurons and 'relu' activation
model.add(Dense(10, activation='softmax'))  # Output layer with 10 neurons for 10 digits and 'softmax' activation

# Compile the model with 'adam' optimizer, 'sparse_categorical_crossentropy' loss, and accuracy as a metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training set for 5 epochs (you can adjust as needed)
model.fit(X_train, y_train, epochs=5)

# Evaluate the model on the testing set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Save the trained model to a file ('digits_model.h5')
model.save('digits_model.h5')
