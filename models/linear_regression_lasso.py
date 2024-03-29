import os
import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.getcwd()

def train_and_evaluate_lasso_regression():
    print("\nLasso Regression Model with Hyperparameter Tuning")
    # Load the dataset
    df = pd.read_csv(os.path.join(BASE_DIR, "database/train.csv"))

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['tweet'])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=[f"{col}_tfid" for col in vectorizer.get_feature_names_out()])

    emotion_encoded = pd.get_dummies(df['emotion'], prefix='emotion')

    X = pd.concat([tfidf_df, emotion_encoded], axis=1)
    y = df['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameters to search over
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'fit_intercept': [True, False]
    }

    # Initialize Lasso regression model
    lasso_model = Lasso()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=lasso_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train, y_train)

    # Print the best parameters
    print("Best parameters found:", grid_search.best_params_)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Train the model with the best hyperparameters
    best_lasso_model = Lasso(**best_params)
    best_lasso_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = best_lasso_model.predict(X_train)
    y_pred_test = best_lasso_model.predict(X_test)

    # Evaluate the model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print("\nTrain Dataset evaluation:")
    print("Mean Squared Error:", mse_train)
    print("Mean Absolute Error:", mae_train)
    print("\n")

    # Load the test dataset
    df_test = pd.read_csv(os.path.join(BASE_DIR, "database/test.csv"))

    X_tfidf_test = vectorizer.transform(df_test['tweet'])
    tfidf_df_test = pd.DataFrame(X_tfidf_test.toarray(), columns=[f"{col}_tfid" for col in vectorizer.get_feature_names_out()])

    emotion_encoded_test = pd.get_dummies(df_test['emotion'], prefix='emotion')

    X_test_final = pd.concat([tfidf_df_test, emotion_encoded_test], axis=1)
    y_test_final = df_test['score']

    # Predict the score using the trained model
    y_pred_test_final = best_lasso_model.predict(X_test_final)

    # Evaluate the model's performance on the test dataset
    mse_test_final = mean_squared_error(y_test_final, y_pred_test_final)
    mae_test_final = mean_absolute_error(y_test_final, y_pred_test_final)

    print("\nTest Dataset evaluation:")
    print("Mean Squared Error:", mse_test_final)
    print("Mean Absolute Error:", mae_test_final)
    print("\n")

    test_df_to_csv = df_test.copy()
    test_df_to_csv['score'] = y_pred_test_final
    print(type(test_df_to_csv))

    output_csv_path = os.path.join(BASE_DIR, 'results/lasso_regression_predictions.csv')

    # Save the merged DataFrame to a CSV file
    test_df_to_csv.to_csv(output_csv_path, index=False)

    eval_df = pd.read_csv(os.path.join(BASE_DIR, 'results/evaluation.csv'))
    new_row = {'algorithm': 'Lasso Regression Model with Hyperparameter Tuning', 'train Mean Squared Error': mse_train, 'train Mean Absolute Error': mae_train, 'test Mean Squared Error': mse_test_final, 'test Mean Absolute Error': mae_test_final}
    eval_df.loc[len(eval_df)] = new_row
    eval_df.reset_index()
    eval_df.to_csv(os.path.join(BASE_DIR, 'results/evaluation.csv'), index=False)


train_and_evaluate_lasso_regression()
