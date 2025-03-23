# Bank Customer Churn Prediction

This repository contains code for predicting customer churn in a banking dataset using a Random Forest Classifier. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Dataset

The dataset used for this project is `BankChurners.csv`, which contains information about bank customers and their churn status.

## Implementation

The implementation includes the following steps:

1.  **Data Loading and Initial Exploration:**
    * Loading the dataset using `pandas`.
    * Basic exploration of the `Customer_Age` and `Gender` columns.
    * Visualizing the age distribution using `seaborn` and `matplotlib`.
    * Visualizing gender distribution and churn rates using pie charts.
    * Visualizing churn rate in various card categories using pie charts.
2.  **Data Preprocessing:**
    * Dropping unnecessary columns (`CLIENTNUM`, `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1`, `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2`).
    * Mapping the `Attrition_Flag` column to binary values (0 and 1).
    * One-hot encoding categorical variables using `pd.get_dummies`.
3.  **Feature Engineering:**
    * Creating new features: `Credit_Utilization_Ratio` and `Avg_Transaction_Amount`.
4.  **Exploratory Data Analysis (EDA):**
    * Generating a correlation matrix heatmap using `seaborn` to visualize feature relationships.
5.  **Model Training:**
    * Splitting the data into training and testing sets using `train_test_split`.
    * Scaling the features using `StandardScaler`.
    * Training a Random Forest Classifier using `RandomForestClassifier`.
6.  **Model Evaluation:**
    * Generating a classification report and confusion matrix using `classification_report` and `confusion_matrix`.
    * Generating a precision-recall curve.
    * Calculating accuracy, F1 score, precision, and recall.

## Files

* `churn_prediction.py`: Python script containing the complete implementation.
* `BankChurners.csv`: The dataset used for this project.

## Dependencies

To run the code, you will need the following Python libraries:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
plotly
```

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly
```

## Usage

1.  Clone the repository:

    ```bash
    git clone https://github.com/ac265640/credit-card-churn
    cd credit-card-churn
    ```

2.  Ensure the `BankChurners.csv` file is in the same directory as the script, or update the `file_path` variable in the script to the correct path.

3.  Run the Python script:

    ```bash
    python churn_prediction.py
    ```

## Code Description

* `load_and_preprocess_data(file_path)`: Loads and preprocesses the dataset.
* `engineer_features(df)`: Creates new features.
* `perform_eda(df)`: Performs exploratory data analysis.
* `train_model(X, y)`: Trains the Random Forest Classifier.
* `evaluate_model(model, X_test, y_test)`: Evaluates the trained model.

## Model Evaluation Metrics

* **Accuracy:** Overall correctness of the model.
* **F1 Score:** Harmonic mean of precision and recall.
* **Precision:** Proportion of correctly predicted positive cases.
* **Recall:** Proportion of actual positive cases that were correctly predicted.
* **Confusion Matrix:** Table showing the performance of the model.
* **Precision-Recall Curve:** Shows the tradeoff between precision and recall.

## Author

Amit Singh Chauhan/ac265640
