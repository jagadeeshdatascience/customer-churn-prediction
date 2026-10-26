# Customer Churn Prediction

## Project Overview
Customer Churn Prediction is a machine learning project that predicts whether a customer is likely to leave a telecommunications company. It uses customer information such as tenure, contract type, internet service, and monthly charges to identify churn patterns.

The project compares multiple classification algorithms and provides an interactive web application built with Streamlit.

## Objectives
- Analyze customer data and identify churn patterns.
- Clean and prepare data for machine learning.
- Train and compare classification models.
- Evaluate model performance using accuracy, precision, recall, and F1-score.
- Build an interactive application to predict customer churn.

## Technologies Used
- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Web Application:** Streamlit
- **Model Saving:** Joblib
- **Version Control:** Git and GitHub

## Machine Learning Models
The following classification algorithms were trained and evaluated:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 80.41% |
| Decision Tree | 79.42% |
| Random Forest | 78.50% |

Logistic Regression achieved the highest accuracy among the three models tested.

A 40% prediction threshold was also tested to improve the identification of customers likely to churn. At this threshold, churn recall was approximately **67%**, compared with 56% at the default 50% threshold.

## Dataset
This project uses the Telco Customer Churn dataset.

**Dataset source:** [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Download the dataset and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project folder before running the training script.

## Project Structure

```text
Customer_Churn_Project/
├── churn_analysis.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset and generated model files are excluded from the Git repository. The training script generates the saved model files when executed.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/jagadeeshdatascience/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Download the dataset

Download the dataset from the Kaggle link above and place the CSV file in the project folder.

### 4. Train the machine learning models

```bash
python churn_analysis.py
```

This script performs data analysis, preprocessing, model training, evaluation, and saves the trained Logistic Regression model and feature columns.

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to use the application.

## Results
- Logistic Regression achieved 80.41% accuracy at the default threshold.
- Lowering the threshold to 40% increased churn recall to approximately 67%.
- The model's coefficients helped identify features associated with higher or lower predicted churn likelihood.
- The Streamlit application allows users to enter customer details and obtain a churn prediction.

These results are based on the current train-test split and should be validated further before real-world use.

## Future Improvements
- Improve churn recall while balancing precision.
- Tune model hyperparameters and prediction thresholds.
- Use cross-validation for more reliable model evaluation.
- Add more visualizations and customer insights.
- Deploy the Streamlit application online.
- Use a preprocessing pipeline to make training and prediction consistent.

## Conclusion
This project demonstrates how machine learning can be used to analyze customer churn, compare classification models, and generate predictions through an interactive application. It combines data analysis, visualization, model evaluation, and application development in one end-to-end project.
