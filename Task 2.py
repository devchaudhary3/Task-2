# --- Task 2: Exploratory Data Analysis (EDA) ---

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target] # Add species names

print("--- Initial Data Inspection ---")
print("\n1. First 5 rows of the DataFrame:")
print(df.head())

print("\n2. DataFrame Information (data types, non-null counts):")
df.info()

print("\n3. Descriptive Statistics for numerical columns:")
print(df.describe())

print("\n4. Check for Missing Values:")
print(df.isnull().sum())

print("\n5. Distribution of Species:")
print(df['species'].value_counts())

print("\n6. Unique values for 'species' column:")
print(df['species'].unique())

print("\n7. Correlation Matrix (Numerical Features):")
# Exclude the 'species' column for correlation calculation
numerical_df = df.drop('species', axis=1)
print(numerical_df.corr())

# You might also want to look at specific value counts for categorical features if any
# (though 'species' is the only 'categorical' in Iris and we've done value_counts)
# print("\nValue counts for 'sepal length (cm)' (example for a numerical feature, less common):")
# print(df['sepal length (cm)'].value_counts().head())