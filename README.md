# 🛰️ Satellite Imagery Based Property Valuation
### Multimodal Machine Learning Project

**Name:** Abhijeet Kumar  
**Enrollment No:** 23324001  
**Project Title:** Satellite Imagery Based Property Valuation  

---

## 📌 Project Overview

Accurate real estate valuation depends on both **structured property attributes** and **environmental context**.  
This project explores whether **satellite imagery** can improve property price prediction when combined with traditional tabular housing data.

A **multimodal regression pipeline** is developed that integrates:
- Numerical housing features (tabular data)
- Satellite images (visual data)

The performance of the multimodal model is rigorously compared against a **strong tabular-only baseline**.

---

## 🎯 Objectives

- Build a **tabular baseline model** for property price prediction
- Programmatically acquire **satellite images** using geographic coordinates
- Design a **multimodal neural network** combining images and tabular data
- Compare performance using **RMSE** and **R²**
- Analyze whether satellite imagery adds predictive value
- Demonstrate scientific rigor through honest evaluation

---

## 📂 Project Structure
Satellite_CDC_Project/
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── images/
│ ├── train/
│ └── test/
│
├── dataset.py
├── multimodal_model.py
├── train_tabular_baseline.py
├── train_multimodal.py
├── evaluate_multimodal.py
├── predict.py
├── requirements.txt
├── README.md
└── 23324001_report.pdf


---

## 📊 Dataset Description

### 🔹 Tabular Features
- `bedrooms`
- `bathrooms`
- `sqft_living`
- `grade`
- `condition`
- `lat`, `long`
- `price` *(target)*

### 🔹 Visual Features
- Satellite images fetched using **Mapbox Static Images API**
- Images resized to **128×128**
- Normalized and processed using a pretrained CNN

---

## 🧠 Methodology

### 1️⃣ Tabular Baseline Model
- **Model:** Random Forest Regressor  
- **Purpose:** Establish a strong benchmark using structured features only

### 2️⃣ Multimodal Model
- **Image Encoder:** Pretrained ResNet-18 (frozen)
- **Tabular Encoder:** Fully connected neural network
- **Fusion:** Concatenation of image and tabular embeddings
- **Output:** Regression head predicting house price

> CNN weights are frozen to enable efficient CPU-based training.

---

## 📈 Evaluation Metrics

- **RMSE (Root Mean Squared Error)**
- **R² Score**

## 🔍 Key Findings

- The **tabular baseline explains most of the variance** in house prices
- Satellite imagery at fixed resolution **did not add useful predictive signal**
- Multimodal model underperformed due to:
  - Weak visual signal
  - Generic pretrained CNN features
  - Increased model complexity

These results emphasize the importance of **baseline comparison** and **critical evaluation**.

---

## 📌 Conclusion

This project demonstrates a complete **end-to-end multimodal machine learning pipeline** for property valuation.  
Experimental results show that **structured housing attributes dominate price prediction**, while satellite imagery introduces noise rather than useful information in this setup.

Negative results are scientifically valuable and reflect real-world modeling challenges.

---

## 🚀 Future Work

- Fine-tuning CNNs on real estate-specific imagery (GPU required)
- Residual modeling (predicting tabular residuals using images)
- Handcrafted visual features (green cover, road density)
- Higher-resolution or multi-temporal satellite imagery

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- PyTorch, Torchvision
- OpenCV
- Matplotlib
- Mapbox Static Images API

---

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
2️⃣ Train tabular baseline
python train_tabular_baseline.py
3️⃣ Train multimodal model
python train_multimodal.py
4️⃣ Evaluate models
python evaluate_multimodal.py
5️⃣ Generate predictions
python predict.py


🏁 Final Note
This project emphasizes:
End-to-end ML engineering
Multimodal learning
Strong baselines
Scientific honesty
It reflects real-world data science practice, not just metric optimization.






