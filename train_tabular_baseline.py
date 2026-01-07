import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "bedrooms", "bathrooms", "sqft_living",
    "grade", "condition", "lat", "long"
]

df = pd.read_csv("data/train.csv")

X = df[FEATURES]
y = df["price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
preds = model.predict(X_val)

rmse = mean_squared_error(y_val, preds, squared=False)
r2 = r2_score(y_val, preds)

print(f"TABULAR BASELINE → RMSE: {rmse:.2f}, R2: {r2:.3f}")
