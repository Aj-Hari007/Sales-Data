# ============================================================
# EDA ADDITIONAL CODE
# ADD THIS AFTER YOUR DATA CLEANING SECTION
# ============================================================

# ============================================================
# STEP — BASIC DATA UNDERSTANDING
# ============================================================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATA TYPES")
print(df.dtypes)

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# ============================================================
# STEP — MISSING VALUES VISUALIZATION
# ============================================================

plt.figure(figsize=(10,6))

sns.heatmap(
    df.isnull(),
    cbar=False,
    cmap='viridis'
)

plt.title("Missing Values Heatmap")

plt.savefig("missing_values_heatmap.png")

plt.show()

# ============================================================
# STEP — HISTOGRAM DISTRIBUTION
# ============================================================

df.hist(
    figsize=(15,12),
    bins=20
)

plt.suptitle("Histogram Distribution")

plt.savefig("histogram_distribution.png")

plt.show()

# ============================================================
# STEP — BOXPLOT FOR OUTLIERS
# ============================================================

plt.figure(figsize=(12,6))

sns.boxplot(
    data=df[numerical_cols]
)

plt.xticks(rotation=90)

plt.title("Boxplot for Numerical Columns")

plt.savefig("boxplot_outliers.png")

plt.show()

# ============================================================
# STEP — CORRELATION ANALYSIS
# ============================================================

correlation_matrix = df.corr(
    numeric_only=True
)

print("\nCORRELATION MATRIX")

print(correlation_matrix)

# ============================================================
# STEP — CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(12,8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.show()

# ============================================================
# STEP — SCATTER PLOT
# ============================================================

if len(numerical_cols) >= 2:

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        x=numerical_cols[0],
        y=numerical_cols[1],
        data=df
    )

    plt.title(
        f"{numerical_cols[0]} vs {numerical_cols[1]}"
    )

    plt.savefig("scatter_plot.png")

    plt.show()

# ============================================================
# STEP — COUNT PLOT
# ============================================================

if len(categorical_cols) > 0:

    plt.figure(figsize=(8,6))

    sns.countplot(
        x=df[categorical_cols[0]]
    )

    plt.xticks(rotation=45)

    plt.title(
        f"Count Plot - {categorical_cols[0]}"
    )

    plt.savefig("count_plot.png")

    plt.show()

# ============================================================
# STEP — GROUP ANALYSIS
# ============================================================

if (
    len(categorical_cols) > 0 and
    len(numerical_cols) > 0
):

    group_analysis = df.groupby(
        categorical_cols[0]
    )[numerical_cols[0]].mean()

    print("\nGROUP ANALYSIS")

    print(group_analysis)

# ============================================================
# STEP — STRONG CORRELATION ANALYSIS
# ============================================================

print("\nSTRONG CORRELATIONS")

corr_pairs = correlation_matrix.unstack()

corr_pairs = corr_pairs.sort_values(
    kind="quicksort"
)

strong_corr = corr_pairs[
    (corr_pairs > 0.7) &
    (corr_pairs < 1.0)
]

print(strong_corr)

# ============================================================
# STEP — GENERATE EDA REPORT
# ============================================================

report = f"""
====================================================
EXPLORATORY DATA ANALYSIS REPORT
====================================================

1. DATASET OVERVIEW
-------------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}

2. DATA CLEANING
----------------
- Missing values handled
- Duplicates removed
- Text standardized
- Outliers removed

3. ANALYSIS PERFORMED
---------------------
- Statistical Summary
- Missing Value Analysis
- Histogram Distribution
- Boxplot Analysis
- Correlation Analysis
- Scatter Plot Analysis
- Count Plot Analysis
- Group Analysis

4. KEY INSIGHTS
----------------
- Strong correlations identified
- Data distributions analyzed
- Outliers detected and removed
- Relationships between variables explored

====================================================
EDA PROJECT COMPLETED SUCCESSFULLY
====================================================
"""

with open(
    "EDA_Report.txt",
    "w"
) as file:

    file.write(report)

print("\nEDA REPORT GENERATED")

# ============================================================
# STEP — DOWNLOAD OUTPUT FILES
# ============================================================

from google.colab import files

files.download("missing_values_heatmap.png")

files.download("histogram_distribution.png")

files.download("boxplot_outliers.png")

files.download("correlation_heatmap.png")

files.download("scatter_plot.png")

files.download("count_plot.png")

files.download("EDA_Report.txt")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n===================================")
print("EDA PROJECT COMPLETED SUCCESSFULLY")
print("===================================")

print("""
Outputs Generated:
1. Missing Values Heatmap
2. Histogram Distribution
3. Boxplot for Outliers
4. Correlation Heatmap
5. Scatter Plot
6. Count Plot
7. EDA Report
""")
