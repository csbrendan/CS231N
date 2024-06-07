import pandas as pd
import json

json_file_path = 'baseline_results.json'

with open(json_file_path, 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)
df = df[['question', 'true_answer', 'predicted_answer']]

csv_file_path = 'human_eval_baseline.csv'

df.to_csv(csv_file_path, index=False)

print("Excel file has been created successfully.")

