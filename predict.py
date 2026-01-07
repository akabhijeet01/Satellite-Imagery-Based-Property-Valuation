import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader

from dataset import PropertyDataset, TABULAR_COLS
from multimodal_model import MultimodalModel

# =====================
# CONFIG
# =====================
DEVICE = "cpu"
BATCH_SIZE = 8
MODEL_PATH = "multimodal_model.pth"
TRAIN_CSV = "data/train.csv"
IMAGE_DIR = "images/train"

# =====================
# LOAD MODEL
# =====================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = MultimodalModel(len(TABULAR_COLS)).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

# =====================
# LOAD DATASET
# =====================
dataset = PropertyDataset(
    csv_path=TRAIN_CSV,
    image_dir=IMAGE_DIR,
    scaler=checkpoint["scaler"],
    train=True
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# PREDICTION LOOP
# =====================
y_true = []
y_pred = []

with torch.no_grad():
    for img, tab, y in loader:
        img, tab = img.to(DEVICE), tab.to(DEVICE)

        preds = model(img, tab)

        # 🔑 FIX: handle batch_size = 1 safely
        y_true.extend(y.view(-1).cpu().numpy())
        y_pred.extend(preds.view(-1).cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =====================
# METRICS
# =====================
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

# =====================
# COMPARISON TABLE
# =====================
results = pd.DataFrame({
    "Model": [
        "Tabular Only",
        "Tabular + Satellite (Multimodal)"
    ],
    "RMSE": [
        150366.05,
        rmse
    ],
    "R2 Score": [
        0.820,
        r2
    ]
})

print("\n📊 MODEL PERFORMANCE COMPARISON")
print(results.to_string(index=False))

