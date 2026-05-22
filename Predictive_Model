import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import os # Added for path and file existence checks
import kagglehub # Added for data download if needed

# --- Start of robustness block for df and cleaned_retail_sales.csv ---
# This block ensures 'df' is loaded and 'cleaned_retail_sales.csv' exists,
# by recreating it if necessary. This handles NameError for df and FileNotFoundError for the CSV.

df_available = False
try:
    # Check if df is already defined in the current session
    _ = df # Attempt to access df; if NameError, it's not defined
    df_available = True
    print("DataFrame 'df' is already available in memory.")
except NameError:
    print("DataFrame 'df' not found in memory. Attempting to load from 'cleaned_retail_sales.csv'...")

if not df_available:
    if os.path.exists("cleaned_retail_sales.csv"):
        try:
            df = pd.read_csv("cleaned_retail_sales.csv")
            df_available = True
            print("DataFrame 'df' loaded successfully from 'cleaned_retail_sales.csv'.")
        except Exception as e:
            print(f"Error loading 'cleaned_retail_sales.csv': {e}. Regenerating data from source...")
            df_available = False # Loading failed, proceed to regenerate
    else:
        print("File 'cleaned_retail_sales.csv' not found. Regenerating data from source...")

if not df_available:
    # These steps are copied from the previous cell 'wOme77RCyNE8' to recreate the cleaned df
    print("Regenerating cleaned data from source...")
    # Ensure kagglehub is installed for data download
    !pip install kagglehub -q

    # Download dataset
    path = kagglehub.dataset_download("mohammadtalib786/retail-sales-dataset")

    # Load raw dataset
    file_path = os.path.join(path, os.listdir(path)[0])
    df = pd.read_csv(file_path)
    print("Raw DataFrame loaded.")

    # Re-identify column types after initial load for cleaning steps
    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include='object').columns

    # Handle missing values - numerical with median
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    # Handle missing values - categorical with mode
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    print("Missing values handled.")

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print("Duplicates removed.")

    # Clean text data
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    print("Text cleaning completed.")

    # Outlier detection (replicating logic from wOme77RCyNE8)
    # wOme77RCyNE8 used numerical_cols[0] which was 'Transaction ID' for outlier detection.
    if len(numerical_cols) > 0:
        target_col_outlier = numerical_cols[0] # This will be 'Transaction ID' based on wOme77RCyNE8
        Q1 = df[target_col_outlier].quantile(0.25)
        Q3 = df[target_col_outlier].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        df = df[
            (df[target_col_outlier] >= lower_limit) &
            (df[target_col_outlier] <= upper_limit)
        ]
        print(f"Outliers removed based on '{target_col_outlier}'.")
    else:
        print("No numerical columns found for outlier removal.")

    # Save the regenerated cleaned data for future use by this cell or subsequent runs
    cleaned_file = "cleaned_retail_sales.csv"
    df.to_csv(cleaned_file, index=False)
    print(f"Cleaned data regenerated and saved to '{cleaned_file}'.")
# --- End of robustness block ---

# STEP 8 — SELECT TARGET COLUMN
# ============================================================

print("\nAVAILABLE COLUMNS:")
print(df.columns)

# ============================================================
# CHANGE TARGET COLUMN HERE
# Example:
# target_column = "Purchased"
# ============================================================

target_column = df.columns[-1]

print("\nTARGET COLUMN:", target_column)

# ============================================================
# STEP 9 — FEATURE SELECTION
# ============================================================

X = df.drop(target_column, axis=1)
y = df[target_column]

# Identify non-numeric columns in X and drop them
non_numeric_cols = X.select_dtypes(include=['object', 'datetime64']).columns
if len(non_numeric_cols) > 0:
    print(f"\nDropping non-numeric columns from features: {list(non_numeric_cols)}")
    X = X.drop(columns=non_numeric_cols)

# ============================================================
# STEP 10 — TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SIZE:", X_train.shape)
print("TESTING DATA SIZE:", X_test.shape)

# ============================================================
# STEP 11 — BUILD MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ============================================================
# STEP 12 — TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

# ============================================================
# STEP 13 — TEST MODEL USING TEST DATASET
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# STEP 14 — ACCURACY SCORE
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:", accuracy)

# ============================================================
# STEP 15 — CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

# ============================================================
# STEP 16 — VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.savefig("confusion_matrix.png")

plt.show()

# ============================================================
# STEP 17 — CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")

print(classification_report(y_test, y_pred))

# ============================================================
# STEP 18 — ROC CURVE
# ============================================================

# Works for binary classification

if len(np.unique(y)) == 2:

    y_prob = model.predict_proba(X_test)[:,1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6,4))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC Score = {auc_score:.2f}"
    )

    plt.plot([0,1], [0,1], linestyle='--')

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.savefig("roc_curve.png")

    plt.show()

# ============================================================
# STEP 19 — FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(feature_df)

# ============================================================
# STEP 20 — VISUALIZE FEATURE IMPORTANCE
# ============================================================

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_df
)

plt.title("Feature Importance")

plt.savefig("feature_importance.png")

plt.show()

# ============================================================
# STEP 21 — SAVE TEST DATASET OUTPUT
# ============================================================

test_output = X_test.copy()

test_output["Actual"] = y_test.values

test_output["Predicted"] = y_pred

test_output.to_csv(
    "test_dataset_predictions.csv",
    index=False
)

print("\nTEST DATASET OUTPUT SAVED")

# ============================================================
# STEP 22 — DOWNLOAD OUTPUT FILES
# ============================================================


files.download("test_dataset_predictions.csv")

files.download("confusion_matrix.png")

if len(np.unique(y)) == 2:
    files.download("roc_curve.png")

files.download("feature_importance.png")

# ============================================================
# STEP 23 — FINAL SUMMARY
# ============================================================

print("\n===================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("===================================")

print("""
Outputs Generated:
1. Test Dataset Predictions CSV
2. Confusion Matrix
3. ROC Curve
4. Feature Importance Chart
5. Accuracy Score
6. Classification Report
""")
