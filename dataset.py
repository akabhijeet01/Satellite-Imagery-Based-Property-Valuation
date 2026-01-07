import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

TABULAR_COLS = [
    "bedrooms", "bathrooms", "sqft_living",
    "grade", "condition"
]

class PropertyDataset(Dataset):
    def __init__(self, csv_path, image_dir, scaler=None, train=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.train = train

        # Tabular preprocessing
        X = self.df[TABULAR_COLS].values
        self.scaler = scaler or StandardScaler()
        self.X_tab = self.scaler.fit_transform(X) if train else self.scaler.transform(X)

        if train:
            self.y = self.df["price"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 🖼️ Load image
        img_path = os.path.join(self.image_dir, f"{int(row['id'])}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 🔽 Resize image (CPU optimization)
        image = cv2.resize(image, (128, 128))

        # 🔢 Normalize & convert to tensor
        image = image / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        # 🔢 Tabular features
        tabular = torch.tensor(self.X_tab[idx], dtype=torch.float32)

        if self.train:
            return image, tabular, torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            return image, tabular

