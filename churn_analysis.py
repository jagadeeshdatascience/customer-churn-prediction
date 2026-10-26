import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display the first five rows
print(data.head())

# Check the number of rows and columns
print(data.shape)

# Check column names and data types
data.info()

# Count churn and non-churn customers
print(data["Churn"].value_counts())

# Display all column names
print("\nColumn Names:")
print(data.columns)

# Numerical statistics
print("\nNumerical Statistics:")
print(data.describe())

# Churn counts
print("\nChurn Counts:")
print(data["Churn"].value_counts())

# Churn percentages
print("\nChurn Percentages:")
print(data["Churn"].value_counts(normalize=True) * 100)

# Missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(data.duplicated().sum())

# Check the data type
print(data["TotalCharges"].dtype)

# Check blank values
print(data[data["TotalCharges"].str.strip() == ""]["TotalCharges"])

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Check missing values
print(data["TotalCharges"].isnull().sum())

print(data[data["TotalCharges"].isnull()])
data.loc[data["TotalCharges"].isnull(), "TotalCharges"] = 0

print(data.info())

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nTotalCharges data type:")
print(data["TotalCharges"].dtype)

print("\nFirst five rows:")
print(data.head())

# Count customers who stayed and left
print(data["Churn"].value_counts())

# Plot churn counts
sns.countplot(data=data, x="Churn")

plt.title("Customer Churn Count")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

sns.histplot(data=data, x="tenure", hue="Churn", bins=20)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.show()

sns.boxplot(data=data, x="Churn", y="MonthlyCharges")

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.show()

sns.countplot(data=data, x="Contract", hue="Churn")

plt.title("Contract Type vs Churn")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
sns.countplot(data=data, x="PaymentMethod", hue="Churn")

plt.title("Payment Method vs Churn")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# Input features
X = data.drop(["Churn", "customerID"], axis=1)

# Target variable
y = data["Churn"]

print("X shape:", X.shape)
print("y shape:", y.shape)

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, drop_first=True, dtype=int)

print(X.head())
print("X shape after encoding:", X.shape)
y = y.map({"No": 0, "Yes": 1})

print(y.head())
print(y.value_counts())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# Verify everything
print("Missing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in X_test:", X_test.isnull().sum().sum())
print("Missing values in y_train:", y_train.isnull().sum())
print("Missing values in y_test:", y_test.isnull().sum())

# Import Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create and train the model
# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

print("Model training completed!")

# Predict churn for test data
y_pred = model.predict(X_test)

print("Actual values:")
print(y_test.head(10).values)

print("Predicted values:")
print(y_pred[:10])

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stayed", "Churned"]
)

display.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Import and train the model
from sklearn.tree import DecisionTreeClassifier

# Create the model
dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
dt_model.fit(X_train, y_train)

print("Decision Tree training completed!")

dt_pred = dt_model.predict(X_test)

# Evaluate the model

from sklearn.metrics import accuracy_score, classification_report

dt_accuracy = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

# Compare both models
print("\nModel Comparison")

print("Logistic Regression:", accuracy * 100, "%")
print("Decision Tree:", dt_accuracy * 100, "%")

# Train a Random Forest model
from sklearn.ensemble import RandomForestClassifier

# Create the model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

print("Random Forest training completed!")

# Make predictions
rf_pred = rf_model.predict(X_test)

# Evaluate accuracy and classification metrics

from sklearn.metrics import accuracy_score, classification_report

rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Compare all three models
print("\nFinal Model Comparison")

print("Logistic Regression:", accuracy * 100, "%")
print("Decision Tree:", dt_accuracy * 100, "%")
print("Random Forest:", rf_accuracy * 100, "%")

# Compare precision, recall, and F1-score
from sklearn.metrics import classification_report

print("\nLogistic Regression Report:")
print(classification_report(y_test, y_pred))

print("\nDecision Tree Report:")
print(classification_report(y_test, dt_pred))

print("\nRandom Forest Report:")
print(classification_report(y_test, rf_pred))

# Improve the Logistic Regression model
# Predict churn using the selected 40% threshold
y_prob = model.predict_proba(X_test)[:, 1]

threshold = 0.40
y_pred_final = (y_prob >= threshold).astype(int)

print("Classification Report at 40% Threshold:")
print(classification_report(y_test, y_pred_final))

# Feature importance using Logistic Regression

feature_names = X_train.columns
coefficients = model.coef_[0]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

importance["Absolute_Value"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    by="Absolute_Value",
    ascending=False
)

print("\nTop 10 Features Associated with Churn:")
print(importance.head(10))

# Plot the results
top_features = importance.head(10).sort_values(
    by="Coefficient"
)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Coefficient"]
)

plt.title("Factors Associated with Customer Churn")
plt.xlabel("Logistic Regression Coefficient")
plt.tight_layout()
plt.show()

import joblib

# Save the trained model
joblib.dump(model, "churn_model.pkl")

# Save the feature columns used during training
joblib.dump(list(X_train.columns), "churn_features.pkl")

print("Model saved successfully!")