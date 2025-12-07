# Customer Churn Prediction

End-to-end ML pipeline for telecom customer churn prediction.

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn category_encoders scipy

# Run notebooks in order
cd notebooks/
# Execute: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

## Project Structure

```
├── notebooks/           # 8 sequential ML pipeline notebooks
├── data/01_raw/        # Cell2Cell dataset
├── data/06_models/     # Trained models (.pkl)
├── data/08_reporting/  # SHAP plots, comparisons
└── docs/               # Full documentation
```

## Documentation

See [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) for:
- Why this project
- Technical decisions & tradeoffs
- Challenges & solutions
- Interview discussion guide

## Notebooks

1. **01_Data_Profiling** - Data quality, missing values
2. **02_EDA** - Statistical tests, correlations
3. **03_Preprocessing** - Imputation, encoding
4. **04_Feature_Engineering** - Domain features
5. **05_Feature_Selection** - Variance/MI/RFE
6. **06_Model_Training** - LR, XGBoost, LightGBM
7. **07_Evaluation** - ROC, PR, Lift charts
8. **08_Interpretation** - SHAP analysis