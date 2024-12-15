# AI-Based Grade Prediction

## Problem Statement

The goal of this project is to build a machine learning model that can predict a student's final grade based on various input features like attendance, participation, assignment scores, and exam marks. This task is crucial as it helps in identifying students who might need extra help before final evaluations.

## Approach

1. **Data Preprocessing**:
    - **Handling Missing Values**: Missing values in numerical columns are filled with the mean of the column, while missing values in categorical columns are replaced by the mode (most frequent value).
    - **One-Hot Encoding**: Categorical features are encoded using one-hot encoding to ensure the model can work with these values.
    - **Feature Scaling**: Numerical features are scaled to bring them within a specific range (0 to 1) using `MinMaxScaler`.

2. **Modeling**:
    - **Decision Tree Classifier** was trained to predict the final grade (`final_grade_U`).
    - **Training and Evaluation**: The dataset is split into training and testing subsets. The model's performance is evaluated using accuracy, classification report, and confusion matrix.

3. **Model Evaluation**:
    - **Confusion Matrix**: This helps visualize the distribution of predicted vs actual class values.
    - **Accuracy and Classification Metrics**: Performance of the model is reported with metrics like accuracy, precision, recall, and F1-score.

## Results

- **Accuracy**: 100% (replaced with actual accuracy achieved by my model)
- **Classification Report**: Detailed metrics including precision, recall, and F1-score.
- **Confusion Matrix**: Visualized the model’s confusion matrix to show the distribution of predicted vs actual values.
- **Feature Importance Analysis**: The Decision Tree Classifier was trained to predict final_grade_U with balanced class weights. Key features such as absences, Walc (weekend alcohol consumption), and goout (social activities) showed strong associations with academic performance. High absenteeism and excessive weekend alcohol consumption negatively impact grades, while a balance between social activities and studies is crucial. These insights suggest that improving attendance, promoting responsible weekend behaviors, and encouraging a healthy social-studies balance can enhance student performance.

## Challenges

1. **Handling Missing Data**: Deciding on the most appropriate way to handle missing data for categorical and numerical columns. This required experimenting with different imputation strategies (mean and mode).
   
2. **Feature Selection**: Selecting the right features to feed into the model while ensuring all relevant data is considered for predictions. Some features were identified as highly correlated with the target.
   
3. **Model Overfitting**: The decision tree model tended to overfit on small datasets. Proper evaluation and tree pruning were implemented to avoid this issue.



## Files

1. **preprocess.py**: Data preprocessing code.
2. **train.py**: Model training and evaluation code.
3. **evaluate_model.py**: Script for confusion matrix and model evaluation.
4. **grade_predictor_model.pkl**: The trained model file.

## Screenshots  

![Preprocessing](<Screenshot 2024-12-15 120927.png>)
- **Training Logs**: Log of model training with key parameters.

-**Feature Importance** A bar chart showing the importance of each feature in predicting the final grade.
![Feauture_analysis-variance](<Screenshot 2024-12-15 130128.png>)

- **Validation Results**: Performance metrics on the validation/test data.
![Training and Evaluating](<Screenshot 2024-12-15 120948.png>)

- **Confusion Matrix**: Visualized confusion matrix to demonstrate model predictions.
![Visualising Confusion Matrix](<Screenshot 2024-12-15 120812.png>)


### Example Execution

- **Preprocessing**: `python preprocess.py`
- **Training**: `python train_model.py`
-**Feature_analysis**: `python feature_analysis.py`
- **Model Evaluation**: `python visualisation.py`
