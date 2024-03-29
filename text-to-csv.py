import pandas as pd
import os

BASE_DIR = os.getcwd()
# Path to the folder containing the text files
train_folder_path = os.path.join(BASE_DIR, 'database/train-set(txt)')

# List all text files in the folder
files = [file for file in os.listdir(train_folder_path) if file.endswith('.txt')]

# Read each text file and concatenate into a single DataFrame
dfs = []
for file in files:
    file_path = os.path.join(train_folder_path, file)
    df = pd.read_csv(file_path, sep='\t', names=['id', 'tweet', 'emotion', 'score'])
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.sort_values(by='id', inplace=True)

# Path to save the merged CSV file
output_csv_path = os.path.join(BASE_DIR, 'database/train.csv')

# Save the merged DataFrame to a CSV file
merged_df.to_csv(output_csv_path, index=False)

print("\nTraining Set has been saved as:", output_csv_path)

# Path to the folder containing the text files
test_folder_path = os.path.join(BASE_DIR, 'database/test-set(txt)')

# List all text files in the folder
files = [file for file in os.listdir(test_folder_path) if file.endswith('.txt')]

# Read each text file and concatenate into a single DataFrame
dfs = []
for file in files:
    file_path = os.path.join(test_folder_path, file)
    df = pd.read_csv(file_path, sep='\t', names=['id', 'tweet', 'emotion', 'score'])
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.sort_values(by='id', inplace=True)

# Path to save the merged CSV file
output_csv_path = os.path.join(BASE_DIR, 'database/test.csv')

# Save the merged DataFrame to a CSV file
merged_df.to_csv(output_csv_path, index=False)

print("Test Set has been saved as:", output_csv_path)
