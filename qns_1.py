from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming you have loaded the kidney disease dataset and preprocessed it
# Replace this part with your actual kidney disease dataset loading and preprocessing
# For example, assuming you have a dataframe 'kidney_data' with features and labels

# Synthetic example (replace with your actual dataset)
data = {
    'age': [25, 62],
    'rc': [4, 5],
    'wc': [6600, 7200],
    'bp': [70, 80],
    'pot': [4.2, 2.5],
    'pcv': [38, 40],
    'class': ['CKD', 'No CKD']
}

kidney_data = pd.DataFrame(data)

# Split the dataset into features (X) and labels (y)
X = kidney_data.drop('class', axis=1)
y = kidney_data['class']

# Create a Decision Tree model
tree_model = DecisionTreeClassifier()

# Train the Decision Tree model
tree_model.fit(X, y)

# Use the trained tree to make predictions for Person A and Person B
person_A = np.array([[25, 4, 6600, 70, 4.2, 38]])  # Replace with actual values
person_B = np.array([[62, 5, 7200, 80, 2.5, 40]])  # Replace with actual values

prediction_A = tree_model.predict(person_A)
prediction_B = tree_model.predict(person_B)

# Print predictions
print("Person A Prediction:", prediction_A)
print("Person B Prediction:", prediction_B)

# Plot and save the decision tree
plt.figure(figsize=(15, 10))
plot_tree(tree_model, filled=True, feature_names=X.columns, class_names=['No CKD', 'CKD'])
plt.savefig('tree.pdf')
