# Satellite_Imagery_Based_Property_Valuation
🛰️ Satellite Imagery Based Property Valuation

Multimodal Machine Learning Project

Name: Abhijeet Kumar
Enrollment No: 23324001
Project Title: Satellite Imagery Based Property Valuation

📌 Project Overview

Accurate real estate valuation depends on both structured property attributes and environmental context.
This project explores whether satellite imagery can improve property price prediction when combined with traditional tabular housing data.

A multimodal regression pipeline was developed that integrates:

Numerical housing features (tabular data)

Satellite images (visual data)

The performance of the multimodal model is rigorously compared against a strong tabular-only baseline.

🎯 Objectives

Build a tabular baseline model for property price prediction

Programmatically acquire satellite images using geographic coordinates

Design a multimodal neural network combining images and tabular data

Compare performance using RMSE and R²

Analyze whether satellite imagery adds predictive value

Demonstrate scientific rigor through honest evaluation

📂 Project Structure
Satellite_CDC_Project/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── images/
│   ├── train/
│   └── test/
│
├── dataset.py                 # Custom PyTorch Dataset
├── multimodal_model.py        # CNN + Tabular fusion model
├── train_tabular_baseline.py  # Random Forest baseline
├── train_multimodal.py        # Multimodal training script
├── evaluate_multimodal.py     # Model comparison & metrics
├── predict.py                 # Test set prediction
├── requirements.txt
├── README.md
└── 23324001_report.pdf        # Final project report

📊 Dataset Description
🔹 Tabular Features

bedrooms

bathrooms

sqft_living

grade

condition

lat, long

price (target)

🔹 Visual Features

Satellite images fetched using Mapbox Static Images API

Images resized to 128×128

Normalized and processed using a pretrained CNN

🧠 Methodology
1️⃣ Tabular Baseline

Model: Random Forest Regressor

Purpose: Establish a strong, reliable benchmark

2️⃣ Multimodal Model

Image Encoder: Pretrained ResNet-18 (frozen, used as feature extractor)

Tabular Encoder: Fully connected neural network

Fusion: Concatenation of image + tabular embeddings

Output: Regression head predicting house price

CNN weights are frozen to enable efficient CPU-based training.

🔍 Key Findings

The tabular baseline explains most of the price variance

Satellite imagery at fixed resolution did not add useful predictive signal

Multimodal model underperformed due to:

Weak visual signal

Generic pretrained CNN features

Increased model complexity

These findings highlight the importance of strong baselines and honest reporting in data science projects.

📌 Conclusion

This project demonstrates an end-to-end multimodal machine learning pipeline for property valuation.
While satellite imagery was hypothesized to improve predictions, experimental results show that structured housing attributes dominate price prediction in this dataset.

Negative results are scientifically valuable and reinforce the importance of rigorous evaluation.

🚀 Future Work

Fine-tuning CNNs on real estate-specific imagery (GPU required)

Residual modeling (predicting tabular residuals using images)

Handcrafted visual features (green cover, road density)

Higher-resolution or multi-temporal satellite images

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

PyTorch, Torchvision

OpenCV

Matplotlib

Mapbox Static Images API

▶️ How to Run
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Train baseline model
python train_tabular_baseline.py

3️⃣ Train multimodal model
python train_multimodal.py

4️⃣ Evaluate models
python evaluate_multimodal.py

5️⃣ Generate predictions
python predict.py

📄 Report

The full project report is available as:

23324001_report.pdf

🏁 Final Note

This project emphasizes:

End-to-end ML engineering

Multimodal learning

Scientific honesty

Strong baseline comparison

It reflects real-world data science practice, not just model optimization.
