import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier

# Load the cleaned dataset
data = pd.read_csv("cleaned_student_data.csv")

# Check for constant features
print("Variance of each feature:")
print(data.var())

# Check for correlations
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()

# Apply one-hot encoding to categorical columns
data = pd.get_dummies(data, drop_first=True)

# Define features and target
target_column = 'final_grade_U'  # Adjust if needed
features = data.drop(columns=[target_column])
target = data[target_column]

# Check target class distribution
print("Target Class Distribution:")
print(target.value_counts())

# Train a Decision Tree model with class weights
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced'
)
model.fit(features, target)

print("Model training completed.")
