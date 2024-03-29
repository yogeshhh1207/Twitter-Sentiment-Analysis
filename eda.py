import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.getcwd()

train_df = pd.read_csv(os.path.join(BASE_DIR, "database/train.csv"))
test_df = pd.read_csv(os.path.join(BASE_DIR, "database/test.csv"))

print('-----------------------------------------------------------------------------------------------------------')
print("\nTraining dataset looks like: \n\n", train_df.head())
print()
print("\nShape of the dataset:", train_df.shape)
print()
print("\nData types and missing values:\n")
print(train_df.info())
print()
print("Emotion and their counts: \n")
print(train_df['emotion'].value_counts())
print('-----------------------------------------------------------------------------------------------------------')
print()
print("\ntest datasets looks like: \n\n", test_df.head())
print()
print("\nShape of the dataset:", test_df.shape)
print()
print("\nData types and missing values:\n")
print(test_df.info())
print('-----------------------------------------------------------------------------------------------------------')