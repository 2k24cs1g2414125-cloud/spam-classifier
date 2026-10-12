# spam-classifier
Machine learning model that classifies SMS messages as spam or not spam using Python, TF-IDF and Naive Bayes.

# Spam Message Classifier

A machine learning project that detects whether an SMS message is spam or not spam.

## How it works
- Cleans the text data and converts it into numbers using TF-IDF
- Trains a Multinomial Naive Bayes model
- Evaluates the model using precision, recall and F1-score

## Tech Stack
Python, pandas, scikit-learn

## Dataset
SMS Spam Collection Dataset (Kaggle)

## How to run
1. Install dependencies: pip install pandas scikit-learn
2. Download spam.csv from Kaggle and keep it in the same folder
3. Run: python spam_classifier.py
