#  1. text-to-csv.py
#  2. evaluation.py
#  linear_regression
#  lasso
#  xgboost_regression

import os
import pandas as pd
import subprocess

def install_packages():
    try:
        # Run the pip install command to install packages from requirements.txt
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
        print("\nAll required packages installed successfully!\n")
    except subprocess.CalledProcessError as e:
        print("\nError occurred while installing packages:", e)

def text_to_csv_data():
    try:
        subprocess.check_call(["python3", "text-to-csv.py"])
        print("\nDataset has been successfully converted to CSV!\n")
    except subprocess.CalledProcessError as e:
        print("\nError occurred while converting dataset to CSV:", e)

def preprocessing():
    try:
        subprocess.check_call(["python3", "preprocessing.py"])
        print("\nDataset has been successfully Preprocessed!\n")
    except subprocess.CalledProcessError as e:
        print("\nError occurred while preprocessing the dataset:", e)

def eda():
    try:
        subprocess.check_call(["python3", "eda.py"])
        print("\nDataset has been successfully Preprocessed!\n")
    except subprocess.CalledProcessError as e:
        print("\nError occurred while preprocessing the dataset:", e)

def initialize_evaluation_csv():
    """
    Initialize an empty evaluation CSV file with column headers.
    """
    BASE_DIR = os.getcwd()

    # Create an empty DataFrame with the specified column headers
    df = pd.DataFrame(columns=['algorithm', 'train Mean Squared Error', 'train Mean Absolute Error',
                               'test Mean Squared Error', 'test Mean Absolute Error'])

    # Define the output CSV file path
    output_csv_path = os.path.join(BASE_DIR, 'results/evaluation.csv')

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)

def linear_regression_ridge():
    try:
        subprocess.check_call(["python3", "models/linear_regression_ridge.py"])
    except subprocess.CalledProcessError as e:
        print("\nError occurred while performing Linear Regression Ridge", e)

def linear_regression_lasso():
    try:
        subprocess.check_call(["python3", "models/linear_regression_lasso.py"])
    except subprocess.CalledProcessError as e:
        print("\nError occurred while performing Linear Regression Lasso", e)

def decision_tree():
    try:
        subprocess.check_call(["python3", "models/decision_tree.py"])
    except subprocess.CalledProcessError as e:
        print("\nError occurred while performing Decision Tree Regression", e)

def random_forest():
    try:
        subprocess.check_call(["python3", "models/randomforest.py"])
    except subprocess.CalledProcessError as e:
        print("\nError occurred while performing Random Forest Regression", e)

def svm():
    try:
        subprocess.check_call(["python3", "models/svm.py"])
    except subprocess.CalledProcessError as e:
        print("\nError occurred while performing Random Forest Regression", e)

def tensorFlow():
    try:
        subprocess.check_call(["python3", "models/tf_keras_regression.py"])
    except subprocess.CalledProcessError as e:
        print("\nError occurred while performing TensorFlow Regression", e)

def evaluate():
    try:
        subprocess.check_call(["python3", "evaluate.py"])
    except subprocess.CalledProcessError as e:
        print("\nError occurred while Evaluating models", e)

if __name__ == "__main__":
    install_packages()
    text_to_csv_data()
    preprocessing()
    eda()
    initialize_evaluation_csv()
    linear_regression_ridge()
    linear_regression_lasso()
    decision_tree()
    random_forest()
    svm()
    tensorFlow()
    evaluate()

