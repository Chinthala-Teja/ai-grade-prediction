import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_evaluate():
    # Load the cleaned dataset
    data = pd.read_csv("cleaned_student_data.csv")
    
    # Print column names to check the correct target column
    print("Column names in the dataset:")
    print(data.columns)
    
    # Update the target column name to match 'final_grade_U'
    target_column = 'final_grade_U'  # Use 'final_grade_U' instead of 'final_grade'
    
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset")

    # Prepare features and target variable
    features = data.drop(columns=[target_column])  # Drop the target column
    target = data[target_column]  # Assign the target column
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Train the Decision Tree Classifier model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    
    # Save the model
    joblib.dump(model, "grade_predictor_model.pkl")
    print("Model saved as grade_predictor_model.pkl")

# Example execution
if __name__ == "__main__":
    train_and_evaluate()
