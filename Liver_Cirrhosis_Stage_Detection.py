#!/usr/bin/env python
# coding: utf-8

# # Liver Cirrhosis Stage Detection

# ## Objective
# Build a system that can output the level of liver damage (liver cirrhosis) given the physical test details of a patient.

# ## Import Libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

import warnings
warnings.filterwarnings('ignore')
print("Libraries loaded successfully.")


# ## Load Dataset

# In[ ]:


# Load dataset
data_path = 'liver_cirrhosis.csv'
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
df.head()


# ## Exploratory Data Analysis & Preprocessing

# In[ ]:


# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Drop non-predictive ID column if it exists
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

# Drop rows with missing target variable
df.dropna(subset=['Stage'], inplace=True)

# Fill other missing values (e.g. median for numerical, mode for categorical)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Features and Target
X = df.drop('Stage', axis=1)
y = df['Stage']

# We need the classes to start from 0 for some models
le_stage = LabelEncoder()
y = le_stage.fit_transform(y)
joblib.dump(le_stage, 'stage_label_encoder.pkl')

print("\nTarget classes:", le_stage.classes_)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")


# ## Model Building

# In[ ]:


# Build a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("Model trained!")


# ## Training the Model

# In[ ]:


# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=[str(c) for c in le_stage.classes_]))

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=le_stage.classes_, yticklabels=le_stage.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Stage')
plt.ylabel('True Stage')
    # plt.show()

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
    # plt.show()


# ## Evaluation

# In[ ]:


# Save model and scaler
joblib.dump(model, 'liver_cirrhosis_model.pkl')
joblib.dump(scaler, 'liver_scaler.pkl')
print("Model and scaler saved.")


# ## Saving Model
