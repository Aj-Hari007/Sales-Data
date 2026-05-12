# -----------------------------
# DATA VISUALIZATION
# -----------------------------

# =====================================================
# DASHBOARD STYLE VISUAL REPORTS
# =====================================================

# ---------- HISTOGRAM ----------

plt.figure(figsize=(8,5))
plt.hist(df[target_col], bins=20)
plt.title(f'Distribution of {target_col}')
plt.xlabel(target_col)
plt.ylabel('Frequency')
plt.show()

# ---------- CORRELATION HEATMAP ----------

plt.figure(figsize=(10,6))

corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True)

plt.title("Correlation Heatmap")
plt.show()

# ---------- COUNT PLOT ----------

if len(categorical_cols) > 0:

    cat_col = categorical_cols[0]

    plt.figure(figsize=(10,5))

    sns.countplot(
        x=df[cat_col],
        order=df[cat_col].value_counts().index
    )

    plt.title(f'Count Plot of {cat_col}')
    plt.xticks(rotation=45)

    plt.show()

# ---------- SCATTER PLOT ----------

if len(numerical_cols) >= 2:

    plt.figure(figsize=(8,5))

    sns.scatterplot(
        x=df[numerical_cols[0]],
        y=df[numerical_cols[1]]
    )

    plt.title(
        f'{numerical_cols[0]} vs {numerical_cols[1]}'
    )

    plt.show()

# ---------- BAR CHART ----------

if len(categorical_cols) > 0 and len(numerical_cols) > 0:

    group_data = df.groupby(cat_col)[target_col].mean()

    plt.figure(figsize=(10,5))

    group_data.plot(kind='bar')

    plt.title(f'Average {target_col} by {cat_col}')

    plt.ylabel(target_col)

    plt.xticks(rotation=45)

    plt.show()

# -----------------------------
# STEP 11: INSIGHTS
# -----------------------------

print("\n==============================")
print("KEY FINDINGS / INSIGHTS")
print("==============================")

print(f"""
1. Dataset cleaned successfully
2. Missing values handled
3. Duplicate rows removed
4. Outliers removed using IQR method
5. Visualizations generated successfully
6. Correlation between numerical features analyzed
""")

# -----------------------------
# STEP 12: SAVE CLEANED DATA
# -----------------------------

cleaned_file = "cleaned_retail_sales.csv"

df.to_csv(cleaned_file, index=False)

print("\nCLEANED DATASET SAVED")

# -----------------------------
# STEP 13: DOWNLOAD CLEANED FILE
# -----------------------------

from google.colab import files

files.download(cleaned_file)

print("\nPROJECT COMPLETED SUCCESSFULLY")
