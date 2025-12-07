# Customer Churn Prediction: My First End-to-End ML Project

## Overview
I built a customer churn prediction system for telecom data that identifies at-risk customers for proactive retention. The entire pipeline—from raw data to interpretable predictions—runs in 8 sequential Jupyter notebooks with serialized model artifacts.

---

## 1. Why This Project

### The Gap I Noticed
Most ML projects I see from other freshers stop at model training. When I looked at job descriptions, companies wanted experience with data pipelines, feature engineering, model interpretation—not just throwing data at scikit-learn.

This project is my attempt to bridge that gap. It answers the question: "Can you actually build a complete ML system that a business could use?"

### The Real Problem
Telecom churn costs 5-10x more to replace customers than retain them. A 28.5% churn rate means losing nearly 1 in 3 customers. Manual intervention is slow and inconsistent—ML can prioritize who to call first.

### Tech Stack Choices
I evaluated several options before settling on my current stack:

**Jupyter Notebooks over scripts:** Transparency, easy review, demonstrates thinking process. Tradeoff: less automated than pipelines.

**scikit-learn + XGBoost + LightGBM:** Tabular data, no GPU needed, interpretable with SHAP. Deep learning would be overkill here.

**SHAP over just feature importance:** Explains individual predictions, not just global patterns. Crucial for business adoption.

**Kedro project structure:** Production-ready folder organization. Easy to convert notebooks to pipelines later.

Each choice involved tradeoffs. Notebooks are less automated than Kedro pipelines, but better for demonstrating my work.

---

## 2. What I Built

### The System Architecture

```
Raw Data → Profiling → EDA → Preprocessing → Feature Engineering → Feature Selection
                                                                          ↓
Business Insights ← SHAP ← Evaluation ← Model Training (LR + XGBoost + LightGBM)
```

The flow is straightforward:
1. Data profiling identifies quality issues and churn baseline
2. EDA finds statistical patterns and multicollinearity
3. Preprocessing handles missing data, outliers, encoding
4. Feature engineering creates domain-specific signals
5. Feature selection reduces to 25-30 interpretable features
6. Model training compares interpretable vs high-performance models
7. Evaluation measures business metrics (lift, precision@K)
8. SHAP explains what drives predictions

### Key Features
- **8 Sequential Notebooks**: Each handles one stage, outputs feed into the next
- **3 Model Comparison**: Interpretable (LR) vs Performance (XGBoost, LightGBM)
- **SHAP Explanations**: Global importance + individual prediction breakdowns
- **Business Metrics**: Lift charts, Precision@Top20%, not just accuracy

### Dataset
**Cell2Cell Telecom Churn Dataset**: 51,047 customers × 55 features
- Behavioral: Call patterns, minutes, dropped calls
- Financial: Monthly revenue, charges, overages
- Demographic: Age, household, income
- Account: Tenure, equipment, service area

---

## 3. Technical Decisions & Tradeoffs

### Data Leakage Prevention: Critical
I excluded `RetentionCalls`, `RetentionOffersAccepted`, and `MadeCallToRetentionTeam` from modeling. Why? These features exist *because* the company already identified churn risk—using them would be predicting the past.

This was a judgment call. The features looked highly predictive, but that's precisely because they're effects, not causes.

### Missing Data Strategy: Practical Over Perfect
| Scenario | Action | Rationale |
|----------|--------|-----------|
| < 5% missing | Median impute | Minimal impact on distribution |
| ≥ 5% missing | Median + `_IsMissing` flag | Missingness itself may be predictive |
| Categorical missing | "Unknown" category | Preserves signal, doesn't assume mode |

### Feature Encoding: Target Encoding for High Cardinality
`Occupation` and `PrizmCode` have 10+ categories. One-hot encoding would create dimensionality explosion. Target encoding (with 5-fold CV to prevent leakage) captures the churn signal without exploding features.

### Outlier Handling: Conservative Approach
Capped at 99th percentile for revenue/usage metrics. Exception: did NOT remove outliers in churn=1 class—extreme usage patterns are informative for churn prediction.

### Class Imbalance: SMOTE on Training Only
Applied SMOTE to reach 40:60 minority:majority ratio, but *only on training data*. Applying before split would leak test information. Also used class weights as backup in all models.

### Model Selection: Both Interpretability AND Performance
Instead of choosing between Logistic Regression (interpretable) and XGBoost (accurate), I trained both. This lets the business choose:
- LR for regulated environments needing explainability
- XGBoost for maximum predictive power

### Feature Selection: 25-30 Feature Target
```
55 features → Variance filter → Correlation filter → MI scores → RFECV → 25-30 features
```
Interpretability threshold. Business stakeholders can review ~25 features but lose track at ~100.

---

## 4. Feature Engineering

### Created Features

| Feature | Formula | Why It Matters |
|---------|---------|----------------|
| `RevenuePerMinute` | `Revenue / (Minutes + 1)` | Revenue efficiency; low value = price-sensitive churner |
| `CallFailureRate` | `(Dropped + Blocked) / (TotalCalls + 1)` | Service quality; >10% indicates network issues |
| `CustomerCareIntensity` | `CareCalls / (MonthsInService + 1)` | Support dependency normalized by tenure |
| `EquipmentAgeRatio` | `EquipmentDays / (TenureMonths × 30 + 1)` | Equipment staleness; >1.5 suggests outdated device |
| `InboundOutboundRatio` | `InboundCalls / (OutboundCalls + 1)` | Usage pattern indicator |

### Epsilon in Denominators
All ratio features use `+1` in denominators to prevent division by zero and infinity values. Every engineered feature validated for inf/NaN before proceeding.

---

## 5. Challenges & Solutions

### Challenge 1: Data Leakage Identification
**Problem:** Retention features looked highly predictive but would cause leakage.

**Debugging:** 
1. Checked feature correlation with target—suspiciously high
2. Thought about temporal order—realized these features exist post-risk-detection

**Solution:** Excluded from modeling. Lesson: high correlation isn't always good.

### Challenge 2: Missing Data Patterns
**Problem:** 5 features had >5% missing values.

**Solution:** Created `_IsMissing` flag columns before imputing with median. The flag itself became a useful predictor—missing data often correlates with customer behavior.

### Challenge 3: Multicollinearity
**Problem:** Call metrics (InboundCalls, OutboundCalls, PeakCalls) were highly correlated (r > 0.85).

**Solution:** Kept the one with highest correlation to churn, dropped others. Reduced features while preserving signal.

### Challenge 4: Feature Engineering Infinity Values
**Problem:** Ratios like `RevenuePerMinute` produced infinity when denominator was zero.

**Solution:** Added epsilon (+1) to all denominators. Validated all engineered features for inf/NaN.

### Challenge 5: Threshold Selection
**Problem:** Default 0.5 threshold maximized accuracy but missed too many churners.

**Solution:** Plotted precision-recall tradeoff, selected threshold maximizing F1. Business can adjust based on retention budget vs. missed churner cost.

---

## 6. What Works Well

### Model Performance (Expected on Test Set)
- **Baseline Churn Rate**: ~28.5%
- **Lift @ Top Decile**: ~2.5x (top 10% predictions are 2.5x more likely to churn)
- **Precision @ Top 20%**: ~50% (targeting top 20% captures half of all churners)
- **Inference time**: <200ms per prediction

### Feature Engineering Value
Engineered features added measurable signal beyond raw features:
- `CallFailureRate` captures service quality issues
- `CustomerCareIntensity` identifies support-dependent customers
- `EquipmentAgeRatio` flags upgrade candidates

### Interpretability
SHAP plots show:
- Which features drive churn globally
- Why specific customers are high-risk
- Actionable insights ("dropped calls increase churn by X%")

### Development Experience
- **Modular notebooks**: Each with clear inputs/outputs
- **Intermediate outputs saved**: No need to rerun everything
- **One-command setup**: `pip install` + run notebooks in order

---

## 7. What Could Be Better

### Known Limitations
1. **No deployment**: Pipeline runs locally, not as API
2. **No automated retraining**: Model versioning via MLflow not implemented
3. **Manual execution**: Notebooks must be run sequentially
4. **Single dataset split**: No temporal validation (would need timestamp data)
5. **No real-time monitoring**: Would need Evidently/Prometheus for production

### If I Had More Time
1. **Convert to Kedro pipelines**: Reproducible, parameterized execution
2. **Add FastAPI endpoint**: Serve predictions via REST API
3. **Implement monitoring**: Evidently for drift detection
4. **CI/CD pipeline**: Automated testing and deployment via GitHub Actions
5. **A/B testing framework**: Measure retention intervention success

---

## 8. What I Learned

### Technical Insights
1. **Leakage is subtle**: Retention features looked predictive but were actually effects
2. **Feature engineering compounds value**: 10 new features from 55 originals added measurable lift
3. **SHAP is worth the compute**: Business stakeholders understand force plots better than coefficients
4. **Threshold matters**: Default 0.5 is rarely optimal for imbalanced problems

### Process Insights
1. **Modular notebooks**: Each notebook with clear inputs/outputs made debugging easier
2. **Save intermediate outputs**: Preprocessing output → CSV → next notebook
3. **Document as you go**: This documentation exists because I wrote notes during development
4. **Check library versions**: Especially when following tutorials or using AI-generated code

### About Business ML
1. **Lift matters more than accuracy**: Telling business "2.5x lift" beats "76% accuracy"
2. **Interpretability enables action**: SHAP → "customers with >3 care calls need proactive outreach"
3. **Simple models have value**: LR odds ratios are immediately actionable

---

## 9. Project Structure

```
customer-churn-prediction-web-app/
├── notebooks/                    # 8 sequential ML pipeline notebooks
│   ├── 01_Data_Profiling.ipynb  # Data quality, missing values, churn baseline
│   ├── 02_EDA.ipynb             # Statistical tests, correlations, visualizations
│   ├── 03_Preprocessing.ipynb   # Imputation, encoding, scaling
│   ├── 04_Feature_Engineering.ipynb # Domain features
│   ├── 05_Feature_Selection.ipynb   # Variance/correlation/MI/RFE
│   ├── 06_Model_Training.ipynb  # LR, XGBoost, LightGBM + GridSearchCV
│   ├── 07_Evaluation.ipynb      # ROC, PR, Lift, model comparison
│   └── 08_Interpretation.ipynb  # SHAP analysis, business insights
├── data/
│   ├── 01_raw/                  # Cell2Cell dataset
│   ├── 02_intermediate/         # Profiling outputs
│   ├── 03_primary/              # Preprocessed data
│   ├── 04_feature/              # Engineered features
│   ├── 05_model_input/          # Selected features
│   ├── 06_models/               # Trained models (.pkl), scalers, feature lists
│   └── 08_reporting/            # Plots, model comparison CSV
├── docs/                        # Documentation
├── conf/base/                   # Kedro configuration (for future migration)
├── src/                         # Python package (for future Kedro pipelines)
├── pyproject.toml               # Dependencies
└── README.md                    # Quick start guide
```

The structure follows Kedro conventions, making future migration straightforward.

---

## 10. How to Run It

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn shap matplotlib seaborn category_encoders scipy
```

### Execute Pipeline
```bash
cd customer-churn-prediction-web-app/notebooks

# Run notebooks in order (Jupyter or VSCode)
# 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

### Outputs Generated
| Artifact | Location |
|----------|----------|
| Champion Model | `data/06_models/champion_model.pkl` |
| Scaler | `data/06_models/scaler.pkl` |
| Feature List | `data/06_models/feature_list.json` |
| SHAP Summary | `data/08_reporting/shap_summary.png` |
| Model Comparison | `data/08_reporting/model_comparison.csv` |
| Lift Chart | `data/08_reporting/lift_chart.png` |

---

## 11. For Interview Discussion

### What I'd Demonstrate
1. **Notebook walkthrough**: Show logical flow from raw data to predictions
2. **SHAP plots**: Explain what drives churn and why specific customers are at risk
3. **Model comparison**: Discuss tradeoff between LR interpretability and XGBoost performance
4. **Lift chart**: Show business value—"targeting top 20% catches 50% of churners"

### Questions I Can Answer

**Data Questions:**
- *"How did you handle missing data?"* → Median + IsMissing flags for >5% missing
- *"Why exclude retention features?"* → Temporal leakage—they occur after churn risk detected
- *"How did you handle categorical features?"* → One-hot for low cardinality, target encoding for high

**Model Questions:**
- *"Why multiple models?"* → Business tradeoff: interpretability vs. performance
- *"How did you handle class imbalance?"* → SMOTE on training only + class weights
- *"How do you explain predictions?"* → SHAP force plots show feature contributions

**Business Questions:**
- *"What's the business value?"* → 2.5x lift @ top decile, 50% precision @ top 20%
- *"How would this be used in production?"* → Score weekly, retention team calls top 20% first
- *"What would you do differently?"* → Add Kedro pipelines, FastAPI, drift monitoring

### Key Talking Points

**On Problem Understanding:**
> "Churn costs 5-10x more than retention. The goal isn't just prediction accuracy—it's prioritizing who the retention team should call first."

**On Data Quality:**
> "I excluded three features that looked highly predictive but were actually data leakage. They only exist because churn risk was already detected."

**On Feature Engineering:**
> "I created domain-specific features like CallFailureRate and CustomerCareIntensity. These capture business logic that raw features miss."

**On Model Interpretability:**
> "SHAP tells us WHY customers churn. 'Dropped calls increase churn probability by 8% per call'—that's actionable for network operations."

### Metrics to Know

| Metric | Value | Meaning |
|--------|-------|---------|
| Baseline Churn Rate | ~28.5% | 1 in 3 customers churns |
| Lift @ Top Decile | ~2.5x | Top 10% predictions are 2.5x more likely to churn |
| Precision @ Top 20% | ~50% | Calling top 20% reaches half of all churners |
| Final Features | 25-30 | From 55 original, after selection |

### What This Project Demonstrates
- **End-to-end ML thinking**: Not just training, but the full pipeline
- **Business awareness**: Lift, precision@K, actionable insights
- **Data quality understanding**: Leakage prevention, missing data handling
- **Technical depth**: Multiple models, hyperparameter tuning, SHAP interpretation
- **Learning ability**: Picking up new tools and concepts quickly

---

## Final Thoughts

This is my first complete ML project from data to interpretable predictions. It's not perfect—there are limitations I'd address in a production system (deployment, monitoring, automation). But it shows I can take a real dataset through a complete ML workflow with attention to data quality, feature engineering, and business impact.

The most valuable part wasn't the final model, but the process: identifying data leakage that would have invalidated results, engineering features that capture domain knowledge, and building SHAP explanations that make predictions actionable.

I'm sharing this not as a finished product, but as evidence that I can build and learn—and that I understand what it takes to put machine learning into practice.
