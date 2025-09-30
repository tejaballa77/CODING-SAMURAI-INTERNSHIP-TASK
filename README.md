# 1. 📊 Telco Customer Churn Prediction

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

2. # 🐦 Twitter Sentiment Analysis

## Project Overview
The Twitter Sentiment Analysis project aims to classify tweets into positive, negative, or neutral sentiments. Understanding public sentiment on social media helps businesses, policymakers, and researchers gauge opinions, track trends, and make informed decisions. This project leverages Natural Language Processing (NLP) and machine learning for sentiment classification.

---

## Dataset
The dataset contains real tweets from users, including their sentiment labels. Some important columns:

- `twitter_id` – Unique ID for each tweet  
- `airline_sentiment` – Target variable (positive, negative, neutral)  
- `text` – The actual tweet text  
- `retweet_count` – Number of times the tweet was retweeted  
- `airline` – Airline related to the tweet  
- Other metadata: `tweet_coord`, `tweet_created`, `tweet_location`, `user_timezone`  

---

## Data Preprocessing
Data cleaning and preprocessing steps performed:

1. **Text cleaning** – Removed punctuation, numbers, URLs, and special characters.  
2. **Lowercasing** – Converted all text to lowercase for uniformity.  
3. **Tokenization** – Split text into individual words (tokens).  
4. **Stopword removal** – Removed common words like “the”, “is”, “and” that do not add meaning.  
5. **Vectorization** – Converted text into numeric format using techniques like TF-IDF.  
6. **Train-test split** – Divided the dataset into training and testing sets for model evaluation.  

---

## Machine Learning Models
Multiple models were trained and evaluated for sentiment classification:

1. **Logistic Regression** – Simple linear model for multi-class classification.  
2. **Multinomial Naive Bayes** – Suitable for text classification and categorical features.  
3. **Support Vector Classifier (SVC)** – Effective for high-dimensional text data.  
4. **Random Forest Classifier** – Ensemble model improving prediction performance.  

### Model Evaluation Metrics
- **Accuracy** – Overall correctness of predictions.  
- **Precision** – How many predicted positives were actually positive.  
- **Recall** – How many actual positives were correctly identified.  
- **F1-score** – Balance between precision and recall.  
- **Confusion Matrix** – Visualization of correct and incorrect predictions for each sentiment class.  

---

## Insights
- Most negative tweets were related to service issues.  
- Positive tweets often contained praise or good experiences.  
- Neutral tweets were informational or general statements.  
- Logistic Regression and Random Forest provided reliable classification performance.  

---

## Skills Learned
- Text preprocessing and cleaning for NLP  
- Feature extraction from text (TF-IDF, Bag of Words)  
- Multi-class classification with machine learning models  
- Evaluating model performance using precision, recall, F1-score, and confusion matrix  
- Drawing actionable insights from social media data  

---

**Hashtags:**  
#DataAnalysis #MachineLearning #SentimentAnalysis #NLP #Python #TextMining #BusinessInsights #TwitterAnalytics


