🔹 European Housing Price Prediction

Math intuition → data → baseline ML (Month 0 project)

A foundation project focused on understanding why ML works — and where it breaks — before using complex models.

🚀 What This Project Proves

How real-world housing data becomes vectors and matrices

Why gradients are essential for learning

How biased datasets silently destroy model reliability

Why simple models fail before complex ones should be used

📊 Key Results (Read This First)
Item	Value
Dataset size	~10,000 rows
Model	Linear Regression
Initial MAE	~€50,000
Final MAE	~€42,500
Relative error	16.7% → 14.2%
Urban vs suburban error gap	+21%

⚠️ Main issue: Urban bias inflates price predictions outside major cities.

🧠 Bias & Failure Analysis (Non-Optional)

This model fails predictably:

Overestimates rural/suburban prices

Error variance increases with price

Performs well only where data density is high

Conclusion:

Model accuracy improved, but generalization did not — due to dataset bias.

This is measured, not ignored.

📂 Project Structure
├── data/                  # Raw housing dataset
├── week1_math/             # NumPy-based math intuition
├── week2_data/             # Data cleaning & baseline ML
├── plots/                  # Exploratory visualizations
└── README.md

🧪 Experiments Summary
Week 1 — Math Intuition

Vector & matrix operations (NumPy)

Gradient approximation

Chain rule intuition

Probability & distributions

📌 Insight:
Vectorization explains why ML scales — Python loops do not.

Week 2 — Real Data & Baseline ML

Missing value handling (mean vs median)

Feature engineering (price_per_sqm)

Linear regression baseline

MAE evaluation

📌 Insight:
Data understanding improved performance more than any algorithm change.

❌ What Did NOT Work (Important)

Linear models fail on extreme prices

Adding features doesn’t fix biased sampling

Complexity without better data is useless

These failures justify moving to tree-based models next.

🔧 Installation
git clone https://github.com/diyor-ai/european-housing-intuition.git
pip install numpy pandas matplotlib seaborn scikit-learn

🔜 Next Steps

Tree-based models (Random Forest, XGBoost)

Better regional balance

Error segmentation by geography

🧠 Final Takeaway

Machine learning doesn’t fail because models are weak —
it fails because data assumptions are wrong.

This project builds the intuition needed before scaling complexity.
