European Housing Price Prediction
From Math Intuition to Baseline Machine Learning
Overview

This project demonstrates why machine learning works with numbers by building a baseline housing price prediction model on European real estate data.

The goal is not to achieve state-of-the-art performance, but to develop strong intuition about:

how real-world data becomes vectors,

how gradients enable learning,

and how biased data breaks models.

This repo is a Month 0 foundation project, preparing for more advanced ML and deep learning work.

Problem Statement

Housing prices in Europe vary significantly by city, size, and local conditions.
Predicting prices is deceptively simple — until real-world issues appear:

missing values,

skewed distributions,

and strong urban bias.

This project answers one core question:

What breaks first when we apply simple ML models to real housing data — and why?

Dataset

Source: Kaggle – European Real Estate Dataset
Rows: ~10,000 (after cleaning)
Scope: Major EU cities

Main features:

city

price

square_meters

bedrooms

⚠️ Known limitation:
The dataset overrepresents large cities (e.g., Paris, Berlin), with limited suburban and rural data.

Bias Analysis

Urban bias significantly affects model behavior.

Observed errors:

MAE (urban data): ~€38,000

MAE (suburban subset): ~€46,000 (+21% increase)

Cause:
City-heavy sampling biases the model toward higher price ranges.

Impact:
The model systematically overestimates prices for non-urban properties.

This bias is measured, not ignored.

Project Structure
├── data/
│   └── europe_housing_prices.csv
├── week1_math/
│   ├── vectors_and_matrices.py
│   ├── gradients_intuition.py
│   └── probability_basics.py
├── week2_data/
│   ├── data_cleaning.py
│   ├── visualization.py
│   └── baseline_model.ipynb
├── plots/
└── README.md

Key Learnings
Math Intuition

Vectors & matrices represent features and transformations

Gradients enable learning via optimization

Chain rule explains how learning propagates through models

Vectorization is why NumPy (and GPUs) are fast

Data & Modeling

Real-world data is noisy and biased

Median imputation outperforms mean for skewed prices

Simple models expose problems early — which is good

Baseline Model

Model: Linear Regression (scikit-learn)

Features:

square meters

bedrooms

city (one-hot encoded)

Metric: Mean Absolute Error (MAE)

Results:

Initial MAE: ~€50,000

After feature engineering & outlier handling: ~€42,500

Relative error reduced from 16.7% → 14.2%

This improvement comes from data understanding, not model complexity.

What Failed (Important)

This project intentionally exposes failures:

Linear regression underestimates expensive properties

Error variance increases with price (heteroscedasticity)

Model generalizes poorly outside major cities

Adding complexity without fixing data does not help

These failures justify moving to more advanced models later.

Installation
git clone https://github.com/yourusername/european-housing-intuition.git
pip install numpy pandas matplotlib seaborn scikit-learn


Place the dataset in the data/ directory and run scripts or notebooks.

Why This Project Matters

Complex ML models are meaningless without understanding:

how data is represented,

where assumptions break,

and how bias distorts predictions.

This project builds that foundation before moving to advanced ML and deep learning.

Next Steps

Non-linear models (tree-based)

Better handling of heteroscedastic errors

Balanced datasets across regions

Transition to advanced ML pipelines

Final Note

This repository prioritizes thinking over performance.
If a simple model fails, the data — not the algorithm — is usually the problem.