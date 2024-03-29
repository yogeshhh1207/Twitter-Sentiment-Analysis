import os
import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

BASE_DIR = os.getcwd()

def train_and_evaluate_cnn_regression():
    print("\nConvolutional Neural Network (CNN) Model for Regression")
    # Load the dataset
    df = pd.read_csv(os.path.join(BASE_DIR, "database/train.csv"))

    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['tweet'])
    tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=[f"{col}_tfid" for col in vectorizer.get_feature_names_out()])

    emotion_encoded = pd.get_dummies(df['emotion'], prefix='emotion')

    X = pd.concat([tfidf_df, emotion_encoded], axis=1)

    y = df['score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vocab_size = X.shape[1]  # Vocabulary size is the number of features after vectorization

    # Define the CNN model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print("\nTrain Dataset evaluation:")
    print("Mean Squared Error:", mse_train)
    print("Mean Absolute Error:", mae_train)
    print("\nTest Dataset evaluation:")
    print("Mean Squared Error:", mse_test)
    print("Mean Absolute Error:", mae_test)

    # Load the test dataset
    df_test = pd.read_csv(os.path.join(BASE_DIR, "database/test.csv"))

    X_tfidf_test = vectorizer.transform(df_test['tweet'])
    tfidf_df_test = pd.DataFrame(X_tfidf_test.toarray(), columns=[f"{col}_tfid" for col in vectorizer.get_feature_names_out()])

    emotion_encoded_test = pd.get_dummies(df_test['emotion'], prefix='emotion')

    X_test_final = pd.concat([tfidf_df_test, emotion_encoded_test], axis=1)
    # Predict the score using the trained model
    y_pred_test_final = model.predict(X_test_final)

    # Evaluate the model's performance on the test dataset
    mse_test_final = mean_squared_error(df_test['score'], y_pred_test_final)
    mae_test_final = mean_absolute_error(df_test['score'], y_pred_test_final)

    print("\nFinal Test Dataset evaluation:")
    print("Mean Squared Error:", mse_test_final)
    print("Mean Absolute Error:", mae_test_final)

    # Save the predictions to a CSV file
    df_test['predicted_score'] = y_pred_test_final
    df_test.to_csv(os.path.join(BASE_DIR, 'results/cnn_regression_predictions.csv'), index=False)

    # Append evaluation metrics to the evaluation CSV file
    eval_df = pd.read_csv(os.path.join(BASE_DIR, 'results/evaluation.csv'))
    new_row = {'algorithm': 'CNN Regression Model', 'train Mean Squared Error': mse_train, 'train Mean Absolute Error': mae_train, 'test Mean Squared Error': mse_test_final, 'test Mean Absolute Error': mae_test_final}
    eval_df.loc[len(eval_df)] = new_row
    eval_df.reset_index()
    eval_df.to_csv(os.path.join(BASE_DIR, 'results/evaluation.csv'), index=False)

train_and_evaluate_cnn_regression()
