import os
import pandas as pd

df = pd.read_csv("data/train.csv")
missing = []

for _ , row in df.iterrows():
    img_path = f"images/train/{int(row['id'])}.png"
    if not os.path.exists(img_path):
        missing.append(row['id'])

print("Missing images:", len(missing))
print(missing[:10])