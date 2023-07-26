# crop-recommendation-s
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 21:02:42 2023

@author: LENOVO
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('D:/Sanket Kakad/archive/Crop_recommendation.csv')

X = df.iloc[:, 3:7]
y = df.iloc[:, 7:8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

A = int(input("Enter the temperature: "))
B = int(input("Enter the humidity: "))
C = int(input("Enter the pH: "))
D = int(input("Enter the rainfall: "))

input_data = [[A, B, C, D]]  
prediction = rf_classifier.predict(input_data)
print("Crop recommendation:", prediction)
