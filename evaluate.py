import pandas as pd
import os

BASE_DIR = os.getcwd()

# Load the dataset
df = pd.read_csv(os.path.join(BASE_DIR, "results/evaluation.csv"))

# Calculate the average performance metrics for each algorithm
df['avg_mse'] = (df['train Mean Squared Error'] + df['test Mean Squared Error']) / 2
df['avg_mae'] = (df['train Mean Absolute Error'] + df['test Mean Absolute Error']) / 2

# Sort the algorithms based on their average performance metrics
df_sorted = df.sort_values(by=['avg_mse', 'avg_mae'], ascending=[True, True])

# Print the rank-wise names of models
print("Rank-wise Names of Models:")
for i, (_, row) in enumerate(df_sorted.iterrows(), start=1):
    print(f"Rank {i}: {row['algorithm']}")

# Display the parameters of the best algorithm
best_algorithm = df_sorted.iloc[0]
print("Best Algorithm:")
print(best_algorithm['algorithm'])
print("Parameters:")
print("Train Mean Squared Error:", best_algorithm['train Mean Squared Error'])
print("Train Mean Absolute Error:", best_algorithm['train Mean Absolute Error'])
print("Test Mean Squared Error:", best_algorithm['test Mean Squared Error'])
print("Test Mean Absolute Error:", best_algorithm['test Mean Absolute Error'])