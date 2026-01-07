print("🚀 train_multimodal.py started")
import torch
torch.set_num_threads(4)

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from dataset import PropertyDataset, TABULAR_COLS
from multimodal_model import MultimodalModel

DEVICE = "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4

train_ds = PropertyDataset("data/train.csv", "images/train", train=True)
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model = MultimodalModel(len(TABULAR_COLS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    losses = []
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for i, (img, tab, y) in enumerate(loader):
        img, tab, y = img.to(DEVICE), tab.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        preds = model(img, tab)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 20 == 0:
            print(f"  Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} mean loss: {np.mean(losses):.4f}")

torch.save({
    "model": model.state_dict(),
    "scaler": train_ds.scaler
}, "multimodal_model.pth")
