import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Subset

from dataset import PropertyDataset, TABULAR_COLS
from multimodal_model import MultimodalModel

print("🚀 evaluate_multimodal.py started")

# =====================
# CONFIG
# =====================
DEVICE = "cpu"
BATCH_SIZE = 8
MODEL_PATH = "multimodal_model.pth"
TRAIN_CSV = "data/train.csv"
IMAGE_DIR = "images/train"
EVAL_FRACTION = 0.2   # evaluate on 20% data

# =====================
# LOAD MODEL
# =====================
print("🔁 Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = MultimodalModel(len(TABULAR_COLS)).to(DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

# =====================
# LOAD DATASET
# =====================
print("📦 Loading dataset...")
full_dataset = PropertyDataset(
    csv_path=TRAIN_CSV,
    image_dir=IMAGE_DIR,
    scaler=checkpoint["scaler"],
    train=True
)

# Use subset for faster evaluation
n_eval = int(len(full_dataset) * EVAL_FRACTION)
indices = list(range(n_eval))
dataset = Subset(full_dataset, indices)

print(f"📊 Evaluating on {n_eval} samples")

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# PREDICTION LOOP
# =====================
y_true = []
y_pred = []

with torch.no_grad():
    for i, batch in enumerate(loader):
        img, tab, y = batch
        img, tab = img.to(DEVICE), tab.to(DEVICE)

        preds = model(img, tab)

        y_true.extend(y.view(-1).cpu().numpy())
        y_pred.extend(preds.view(-1).cpu().numpy())

        if i % 50 == 0:
            print(f"  Processed batch {i}/{len(loader)}")

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

