import os
import pandas as pd

BASE_DIR = os.getcwd()

df = pd.DataFrame(columns=['algorithm', 'train Mean Squared Error', 'train Mean Absolute Error','test Mean Squared Error', 'test Mean Absolute Error'])

output_csv_path = os.path.join(BASE_DIR, 'results/evaluation.csv')

df.to_csv(output_csv_path, index=False)

