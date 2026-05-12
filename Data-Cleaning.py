# =========================================================
# RETAIL SALES DATA CLEANING & VISUALIZATION PROJECT
# Google Colab Runnable Code
# =========================================================

# -----------------------------
# STEP 1: INSTALL & IMPORT
# -----------------------------

!pip install kagglehub -q

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Optional settings
plt.rcParams['figure.figsize'] = (10,5)
sns.set_style("whitegrid")

# -----------------------------
# STEP 2: DOWNLOAD DATASET
# -----------------------------

path = kagglehub.dataset_download(
    "mohammadtalib786/retail-sales-dataset"
)

print("Dataset Path:", path)

# -----------------------------
# STEP 3: LOAD DATASET
# -----------------------------

import os

# Check files inside dataset folder
print(os.listdir(path))

# Replace filename if needed
file_path = os.path.join(path, os.listdir(path)[0])

df = pd.read_csv(file_path)

# -----------------------------
# STEP 4: BASIC DATA ANALYSIS
# -----------------------------

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET INFO")
print(df.info())

print("\nSHAPE OF DATASET")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# -----------------------------
# STEP 5: CHECK MISSING VALUES
# -----------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())

# -----------------------------
# STEP 6: HANDLE MISSING VALUES
# -----------------------------

# Fill numerical columns with median
numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include='object').columns

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMISSING VALUES AFTER CLEANING")
print(df.isnull().sum())

# -----------------------------
# STEP 7: REMOVE DUPLICATES
# -----------------------------

print("\nDUPLICATE ROWS:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("DUPLICATES REMOVED")

# -----------------------------
# STEP 8: CLEAN TEXT DATA
# -----------------------------

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

print("\nTEXT CLEANING COMPLETED")

# -----------------------------
# STEP 9: OUTLIER DETECTION
# -----------------------------

# Select one important numerical column
# Change column name if necessary

print("\nNUMERICAL COLUMNS:")
print(numerical_cols)

# Example target column
target_col = numerical_cols[0]

print("\nUSING COLUMN FOR OUTLIER DETECTION:", target_col)

# Boxplot before removing outliers
plt.figure(figsize=(8,4))
sns.boxplot(x=df[target_col])
plt.title(f'Boxplot Before Outlier Removal - {target_col}')
plt.show()

# IQR METHOD
Q1 = df[target_col].quantile(0.25)
Q3 = df[target_col].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# Remove outliers
df = df[
    (df[target_col] >= lower_limit) &
    (df[target_col] <= upper_limit)
]

print("\nOUTLIERS REMOVED")

# Boxplot after removing outliers
plt.figure(figsize=(8,4))
sns.boxplot(x=df[target_col])
plt.title(f'Boxplot After Outlier Removal - {target_col}')
plt.show()
