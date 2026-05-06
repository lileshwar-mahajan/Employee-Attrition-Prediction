# Employee-Attrition-Prediction
Machine learning pipeline predicting employee attrition from HR data. Gradient Boosting achieved ROC-AUC of 0.9808 across 14,999 employee records with SMOTE balancing, feature engineering, and actionable retention insights.


# 🧠 Employee Attrition Prediction

## 📊 Project Overview

This project presents a comprehensive **Employee Attrition Prediction Pipeline** built using **Python & Scikit-learn**. The pipeline analyzes HR records to identify employees at risk of leaving, compares four classification models, and translates predictions into actionable retention recommendations for HR teams.

## 🎯 Problem Statement

Employee attrition is one of the most costly challenges in modern organisations — replacement costs can reach 50%–200% of annual salary. This analysis aims to answer:

- Which employees are most at risk of leaving?
- What are the key drivers behind attrition?
- Which ML model best identifies flight-risk employees?
- How can HR teams proactively intervene before a resignation?

---
## 🔍 Key Business Questions Answered

1. **Which employees are statistically most likely to leave next month?**
2. **What is the #1 predictor of attrition in the dataset?**
3. **Does overwork or dissatisfaction drive more departures?**
4. **Which departments and salary bands have the highest flight risk?**
5. **How can we deploy this model as a monthly HR risk-scoring tool?**

---

## 📊 Key Results & Insights

### 🏆 Model Performance Summary

| Rank | Model | Accuracy | F1 Score | ROC-AUC | CV-AUC |
|------|-------|----------|----------|---------|--------|
| 1 | **Gradient Boosting** | 0.9808 | **0.9416** | **0.9845** | 0.9970 |
| 2 | Random Forest | 0.9825 | 0.9459 | 0.9805 | 0.9967 |
| 3 | Decision Tree | 0.9416 | 0.8430 | 0.9668 | 0.9819 |
| 4 | Logistic Regression | 0.8003 | 0.5846 | 0.8635 | 0.8632 |

> ✅ **Best Model: Gradient Boosting** — Selected by ROC-AUC for deployment.

---

### 📌 Top Feature Importances (Gradient Boosting)

| Rank | Feature | Importance Score | Business Meaning |
|------|---------|-----------------|-----------------|
| 1 | `satisfactoryLevel` | ~0.41 | Single strongest predictor |
| 2 | `timeSpent.company` | ~0.37 | Tenure-based disengagement |
| 3 | `lastEvaluation` | ~0.10 | Performance vs. recognition gap |
| 4 | `numberOfProjects` | ~0.05 | Workload overload signal |
| 5 | `avgMonthlyHours` | ~0.04 | Burnout indicator |
| 6 | `hours_per_project` | ~0.03 | Engineered: workload intensity |

---

### 🔑 Key EDA Findings

- **16.6% overall attrition rate** — significant class imbalance requiring SMOTE correction
- **Satisfaction < 0.40** — the single sharpest dividing line between leavers and stayers
- **Two burnout clusters** — moderate overwork (125–160 hrs/month) AND extreme overwork (>250 hrs/month) both show elevated attrition
- **Low-salary employees leave at 20.5%** vs only **4.8% for high-salary** peers
- **HR department** has the highest attrition rate (18.8%); **Management** the lowest (11.9%)

---

### 🛠️ Feature Engineering (3 Derived Features)

| Feature | Formula | Business Signal |
|---------|---------|----------------|
| `hours_per_project` | `avgMonthlyHours / numberOfProjects` | High = each project is too demanding |
| `senior_no_promo` | `timeSpent > 5 AND promotion = 0` | Stagnation & disengagement |
| `high_eval_low_sal` | `lastEvaluation > 0.75 AND salary = low` | Top performer underpaid — flight risk |

> All three engineered features appear in the **Top 12 importance rankings**.

---

## 🏗️ System Architecture

```
Input Data (14,999 Employee Records)
            │
            ▼
Preprocessing + Feature Engineering
  • Missing value imputation
  • Duplicate removal
  • Ordinal encoding (salary)
  • One-hot encoding (department)
  • 3 derived features added
            │
            ▼
      SMOTE Class Balancing
  (Training set only — 83.4% Stayed / 16.6% Left)
            │
            ▼
┌──────────────────────────────────────────────┐
│  Logistic  │  Decision  │  Random  │Gradient │
│ Regression │    Tree    │  Forest  │Boosting │
└──────────────────────────────────────────────┘
            │
            ▼
  Model Evaluation & Selection
  (Accuracy, F1, ROC-AUC, 5-Fold CV-AUC)
            │
            ▼
  Attrition Risk Score + Retention Signals
```

---

## 🛠️ Tools & Technologies Used

- **Python 3.x** — Core programming language
- **Scikit-learn** — ML models, SMOTE, cross-validation
- **Pandas & NumPy** — Data manipulation and analysis
- **Matplotlib & Seaborn** — EDA visualisations
- **imbalanced-learn** — SMOTE class balancing
- **Jupyter Notebook** — Development environment

---

## 🚀 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/lileshwar-mahajan/Employee-Attrition-Prediction.git
cd Employee-Attrition-Prediction
```

### 2. Install Required Libraries

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook notebooks/people_analysis_clean.ipynb
```

### 4. Explore the Pipeline

The notebook walks through:
- ✅ Data loading & cleaning
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature engineering
- ✅ SMOTE class balancing
- ✅ Model training (4 classifiers with 5-fold CV)
- ✅ Model evaluation & comparison
- ✅ Feature importance analysis
- ✅ Business recommendations

---

## 📐 ML Pipeline Reference

### Preprocessing Steps

1. **Load dataset** — 14,999 rows, 10 raw features
2. **Impute missing values** — median for numeric, mode for categorical
3. **Remove duplicates** — clean dataset integrity check
4. **Encode salary** — ordinal: `low=0, medium=1, high=2`
5. **One-hot encode department** — 10 department columns
6. **Derive 3 engineered features** — hours_per_project, senior_no_promo, high_eval_low_sal
7. **Train/test split** — 80/20 stratified split
8. **Apply SMOTE** — on training set only to prevent data leakage

### Model Configurations

```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, subsample=0.8, random_state=42
    )
}
```

---

## 📌 Key Risk Factors & Recommendations

### ⚠️ Key Risk Factors

| Signal | Threshold | Action |
|--------|-----------|--------|
| Low satisfaction | < 0.40 | Immediate manager 1:1 + survey |
| Overwork burnout | > 250 hrs/month | Auto manager check-in flag |
| High performer, low pay | Eval > 0.75 + salary = low | Compensation review |
| Senior, no promotion | Tenure > 5 + promotion = 0 | Structured career pathway |
| Project overload | > 6 concurrent projects | Workload rebalancing |

### ✅ HR Recommendations

1. **Quarterly satisfaction surveys** with mandatory follow-up action plans
2. **Soft cap of 4–5 concurrent projects** per employee to prevent overload
3. **Salary band review** for high performers stuck in the low pay bracket
4. **Structured promotion pathways** for employees past year 4 of tenure
5. **Automated alerts** for any employee exceeding 250 hrs/month
6. **Monthly attrition risk scores** surfaced to HR business partners

---

## 📊 Dataset Information

- **Source**: HR Analytics (Kaggle-style synthetic dataset)
- **Total Records**: 14,999 employees
- **Target Variable**: `left` (1 = attrited, 0 = stayed)
- **Class Split**: 83.4% Stayed / 16.6% Left
- **Departments**: 10 (HR, Sales, Technical, Support, Accounting, Marketing, IT, Product_mng, RandD, Management)

### Raw Features

| Feature | Type | Description |
|---------|------|-------------|
| `satisfactoryLevel` | Float | Employee satisfaction score (0–1) |
| `lastEvaluation` | Float | Most recent performance evaluation (0–1) |
| `numberOfProjects` | Integer | Number of projects assigned |
| `avgMonthlyHours` | Integer | Average monthly working hours |
| `timeSpent.company` | Integer | Years at company |
| `workAccident` | Binary | Whether employee had a workplace accident |
| `promotionInLast5years` | Binary | Promoted in past 5 years |
| `salary` | Ordinal | Salary band (low/medium/high) |
| `department` | Categorical | Department name |
| `left` | Binary | **Target** — 1 if employee left |

---

## 🔍 EDA Highlights

### Attrition Split
- **83.4%** Stayed | **16.6%** Left
- SMOTE applied on training set only to generate synthetic minority samples

### Satisfaction vs. Attrition
- Employees who left cluster sharply **below 0.40 satisfaction**
- Strongest single predictor in the dataset

### Salary Level vs. Attrition Rate
- **Low salary**: 20.5% attrition rate
- **Medium salary**: 14.6% attrition rate
- **High salary**: 4.8% attrition rate

### Monthly Hours — Two Burnout Clusters
- **Cluster 1**: 125–160 hrs/month (moderate overwork + dissatisfaction)
- **Cluster 2**: >250 hrs/month (extreme burnout)

---

## 💡 Practical Implications

- Provides a **monthly attrition risk score** for every active employee
- Enables HR teams to **focus retention budgets** on highest-risk individuals
- **Interpretable outputs** — feature importance allows managers to understand *why* an employee is flagged
- Model can be **serialised (pickle/joblib)** and integrated into existing HRIS platforms
- Scalable across different **business units, geographies, or companies** with minimal retraining
- Can be extended to **portfolio-level workforce planning** and succession management

---

## ⚠️ Limitations & Future Scope

### Limitations
- Relies on historical data — may not capture sudden org changes or market shocks
- External factors (economic conditions, competitor offers) are not included
- Dataset is synthetic — real HR data would yield richer signals
- No temporal modelling — satisfaction *trajectory* over time is not captured
- Model does not account for cost asymmetry between False Negatives and False Positives

### Future Scope
- Integrate real-time HRIS data for dynamic monthly risk updates
- Apply **SHAP** for per-employee, locally interpretable risk explanations
- Incorporate **sentiment analysis** from internal surveys and communication tools
- Use **LSTM or Transformer** models to capture satisfaction trends over time
- Extend to automated retention workflow triggers

---

## 🎓 Key Learnings

### Technical Skills Demonstrated
- End-to-end ML pipeline design and implementation
- Handling class imbalance with SMOTE (imbalanced-learn)
- Hyperparameter configuration for ensemble models
- Cross-validation strategy (5-fold stratified CV)
- Feature engineering from business domain knowledge
- Model evaluation using multiple metrics (Accuracy, F1, ROC-AUC)
- Feature importance interpretation (Gini impurity scores)

### Analytical Insights
- Correlation between satisfaction, workload, and attrition
- Non-linear patterns captured by tree-based ensembles
- Importance of domain-driven feature engineering
- Business translation of model outputs into HR action plans

---

## 📧 Contact

For any questions or feedback regarding this project, please feel free to reach out!

**Lileshwar Mahajan**

- 🐙 GitHub: [@lileshwar-mahajan](https://github.com/lileshwar-mahajan)
- 💼 LinkedIn: [Lileshwar Mahajan](https://www.linkedin.com/in/lileshwar-mahajan-b81137255)
- 📧 Email: mahajanlileshwar@gmail.com

---

## ⭐ Acknowledgments

- Kaggle for providing the HR Analytics dataset inspiration
- Scikit-learn and imbalanced-learn communities for open-source tools
- HR analytics practitioners for domain knowledge and benchmarks

**If you find this project helpful, please consider giving it a ⭐ and sharing it with others!**

---

## 🔗 Related Projects

Check out my other data analysis projects:

- [📸 Instagram Analytics Dashboard](https://github.com/lileshwar-mahajan/Instagram-Analytics-Dashboard)
- [🎵 Spotify Talent & Popularity Analysis](https://github.com/lileshwar-mahajan/Spotify-Data-Analysis)
- [🛍️ FNP Sales Analysis](https://github.com/lileshwar-mahajan/FNP-Sales-Analysis)

---

*Last Updated: May 2026*
