import pandas as pd
import zipfile

# Step 1: Unzip the file
with zipfile.ZipFile("train.csv.zip", 'r') as zip_ref:
    zip_ref.extractall()

# Step 2: Load the CSV
df = pd.read_csv("train.csv")

# Step 3: Convert to JSON
json_data = df.to_json(orient='records', lines=True)

# Step 4: Save to a file
with open("train_data.json", "w") as f:
    f.write(json_data)

print("CSV successfully converted to JSON!")
