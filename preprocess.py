import pandas as pd

def preprocess_dataset(input_file, output_file):
    # Load the dataset
    data = pd.read_csv(input_file)

    # Display the first few rows of the dataset (optional, for inspection)
    print("First 5 rows of the dataset:")
    print(data.head())

    # Handle missing values for numerical columns by filling with the column mean
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # Handle missing values for categorical columns by filling with the mode (most frequent value)
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = data[column].fillna(data[column].mode()[0])

    # Apply One-Hot Encoding for categorical columns (after handling missing values)
    data = pd.get_dummies(data, drop_first=True)  # drop_first=True to avoid multicollinearity

    # Save the cleaned dataset to a new CSV file
    data.to_csv(output_file, index=False)

    print(f"Dataset preprocessed and saved at {output_file}")
    print(f"Data Types after preprocessing:\n{data.dtypes}")

if __name__ == "__main__":
    # Define input and output file paths
    input_file = 'student_data.csv'  # Replace with your actual file path
    output_file = 'cleaned_student_data.csv'  # This will be the output file with preprocessed data

    # Preprocess the dataset
    preprocess_dataset(input_file, output_file)
