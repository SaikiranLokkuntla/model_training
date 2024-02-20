# from fastapi import FastAPI

# app = FastAPI()


# @app.get("/")
# sync def root():
 #   return {"message": "Hello World"}


# train_model.py
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from joblib import dump
# Load dataset
data = pd.read_csv('dataset.csv')
X = data.drop('label', axis=1)
y = data['label']
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
# Save model
dump(model, 'model.joblib')
