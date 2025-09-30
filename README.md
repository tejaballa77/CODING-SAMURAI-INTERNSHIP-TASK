# 📊 Telco Customer Churn Prediction

## Project Overview
The Telco Customer Churn Prediction project aims to predict whether a customer will leave a telecom service provider. Predicting churn helps businesses retain customers, optimize marketing strategies, and improve overall customer satisfaction. This project leverages machine learning to classify customers into churned and non-churned categories.

---

## Dataset
The dataset contains customer-level information collected by a telecom company. It includes demographics, account information, and service usage details. Some important columns:

- `gender` – Male/Female  
- `SeniorCitizen` – Indicates if the customer is a senior citizen (0 or 1)  
- `tenure` – Number of months the customer has stayed with the company  
- `MonthlyCharges` – The amount charged to the customer monthly  
- `TotalCharges` – Total amount charged to the customer  
- `HasInternetService` – Whether the customer has internet service (Yes/No)  
- `Churn` – Target variable (Yes = churned, No = not churned)

---

## Data Preprocessing
Data cleaning and preprocessing steps performed:

1. **Handling missing values** – Checked for and removed or imputed any missing data.  
2. **Encoding categorical variables** – Converted categorical features like gender, internet service to numeric values.  
3. **Feature scaling** – Standardized numerical columns to improve model performance.  
4. **Train-test split** – Divided the data into training and testing sets to evaluate model performance.

---

## Machine Learning Models
Multiple classification models were trained and evaluated:

1. **Logistic Regression** – A simple linear model for binary classification.  
2. **Multinomial Naive Bayes** – Probabilistic classifier suitable for categorical features.  
3. **Support Vector Classifier (SVC)** – Works well for high-dimensional feature spaces.  
4. **Random Forest Classifier** – Ensemble model using multiple decision trees to improve accuracy.  

### Model Evaluation Metrics
- **Accuracy** – Overall correctness of the model.  
- **Precision** – How many predicted churns were actually churns.  
- **Recall** – How many actual churns were correctly predicted.  
- **F1-score** – Weighted balance between precision and recall.  
- **Confusion Matrix** – Visual representation of correct and incorrect predictions across classes.  

---

## Insights
- Customers with longer tenure and higher engagement tend to stay.  
- Senior citizens are less likely to churn compared to younger customers.  
- Internet service and monthly charges are significant predictors of churn.  
- Logistic Regression and Random Forest models gave the best balance of accuracy and interpretability.  

---

## Skills Learned
- Data preprocessing and feature engineering  
- Handling categorical and numerical data  
- Implementing and comparing multiple machine learning models  
- Evaluating model performance using classification metrics  
- Drawing business insights from model outputs

---

**Hashtags:**  
#DataAnalysis #MachineLearning #ChurnPrediction #BusinessAnalytics #Python #Classification #CustomerRetention

## How to Run the App
1. Clone the repository:
```bash
git clone https://github.com/your-username/Telco_Churn_App.git

