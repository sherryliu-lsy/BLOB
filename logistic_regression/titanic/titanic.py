# Step 1: Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load the dataset
# Load data from the CSV file
data = pd.read_csv('dataset.csv')

# Inspect the DataFrame
print(data.head())  # Show the first few rows of the DataFrame
print(data.columns)  # List all columns in the DataFrame

# Step 3: Clean the column names by stripping spaces and converting to lowercase
data.columns = data.columns.str.strip().str.lower()

# Define the feature set (X) and the target variable (y)
# Updated to match new column names
X = data[['sex', 'age', 'fare', 'pclass_1', 'pclass_2', 'pclass_3', 'family_size', 'mr', 'mrs', 'master', 'miss', 'emb_1', 'emb_2', 'emb_3']]
y = data['survived']

# Step 4: Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Print the model coefficients and intercept
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model's performance
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Step 9: Visualize the Confusion Matrix using Matplotlib
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix using matplotlib.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add numerical values inside the matrix cells
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f'{cm[i, j]:d}',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Step 10: Call the function to plot the confusion matrix
class_names = ['Did Not Survive', 'Survived']
plot_confusion_matrix(conf_matrix, class_names)
