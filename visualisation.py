import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Load the cleaned dataset
data = pd.read_csv("cleaned_student_data.csv")

# Debug: Print column names to check if 'final_grade_U' exists
print("Column names in the dataset:")
print(data.columns)

# Ensure 'final_grade_U' exists before dropping
if 'final_grade_U' in data.columns:
    features = data.drop(columns=['final_grade_U'])
    target = data['final_grade_U']
else:
    print("Error: 'final_grade_U' column not found!")
    exit()

# Check the class distribution in the target column
print("\nClass distribution in target variable 'final_grade_U':")
print(target.value_counts())

# Split the data into train and test sets (no stratification due to small dataset)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=42)

# Load the pre-trained model
model = joblib.load("grade_predictor_model.pkl")

# Get predictions
predictions = model.predict(X_test)

# Debugging: Print unique values in y_test and predictions to check class distribution
print("\nUnique values in y_test:", np.unique(y_test))
print("Unique values in predictions:", np.unique(predictions))

# Calculate confusion matrix and specify labels (to handle single class cases)
cm = confusion_matrix(y_test, predictions, labels=np.unique(target))

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(target), yticklabels=np.unique(target))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
