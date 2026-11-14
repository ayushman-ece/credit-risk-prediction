# 💳 Credit Risk Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-orange)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Model-red?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview

The Credit Risk Prediction System is a Python-based Machine Learning project that predicts loan approval outcomes using applicant financial information.

It uses a Random Forest Classifier to learn patterns from sample loan application data and classify applications as approved or not approved.

The project includes data validation, exploratory analysis, model training, evaluation, prediction, and CSV report generation.

## 🎯 Objectives

- Analyze loan application data.
- Understand classification problems.
- Train a Random Forest classification model.
- Predict loan approval outcomes.
- Evaluate performance using classification metrics.
- Visualize model results.
- Export predictions into CSV.

## ⚙️ Features

- CSV dataset loading and validation.
- Loan application analysis.
- Approval distribution analysis.
- Train-test splitting with stratification.
- Random Forest Classification.
- Accuracy, Precision, Recall and F1 Score.
- Classification report.
- Confusion matrix visualization.
- New loan application prediction.
- CSV prediction report.

## 🧠 Machine Learning Workflow

1. Load and validate loan application data.
2. Analyze financial information.
3. Separate input features and target variable.
4. Split data into training and testing sets.
5. Train the Random Forest Classifier.
6. Evaluate classification performance.
7. Predict a new loan application.
8. Export prediction results.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning |
| Matplotlib | Visualization |
| Random Forest | Classification |

## 📊 Dataset Information

The sample dataset contains 40 synthetic loan applications.

| Feature | Description |
|---|---|
| applicant_id | Unique applicant identifier |
| annual_income | Annual income |
| credit_score | Applicant credit score |
| loan_amount | Requested loan amount |
| employment_years | Years of employment |
| existing_loans | Number of existing loans |
| loan_approved | Approval outcome (0 or 1) |

**Target variable:** `loan_approved`

- 1 = Approved
- 0 = Not Approved

## 📈 Evaluation Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** How many predicted approvals were actually approved.
- **Recall:** How many actual approvals were correctly identified.
- **F1 Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect classifications.

## 🖥️ Example Prediction

```python
income = 750000
credit_score = 740
loan_amount = 300000
employment_years = 6
existing_loans = 1
```

The model predicts an approval class and estimates an approval probability.

## 📁 Generated Files

| File | Description |
|---|---|
| confusion_matrix.png | Classification visualization |
| loan_predictions.csv | Actual and predicted outcomes |

## 🚀 Installation and Execution

**1. Clone the repository**

```bash
git clone https://github.com/ayushman-ece/credit-risk-prediction.git
```

**2. Open the project folder**

```bash
cd credit-risk-prediction
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate it on Windows**

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the project**

```bash
python main.py
```

## 📚 Key Concepts Learned

- Machine Learning Classification
- Random Forest Classifier
- Train-test splitting
- Stratified sampling
- Accuracy, Precision and Recall
- F1 Score
- Confusion Matrix
- Model prediction
- Data validation
- CSV report generation

## ⚠️ Limitations

- Uses only 40 synthetic sample records.
- Does not represent real banking or lending data.
- Model evaluation is not sufficient to establish real-world performance.
- Does not account for all financial, regulatory, or personal factors.
- Must not be used for actual lending decisions.

## 🔮 Future Improvements

- Use a larger, real-world and appropriately authorized dataset.
- Handle missing values and outliers.
- Compare Logistic Regression, Decision Trees and Random Forest.
- Perform hyperparameter tuning and cross-validation.
- Add feature importance analysis.
- Build a Streamlit dashboard.

## 👨‍💻 Author

**Ayushman Tiwari**

GitHub: [@ayushman-ece](https://github.com/ayushman-ece)

---

⭐ If you find this project useful, consider giving the repository a star!