# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For visualization

# Ignore warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore')

# Load dataset (Make sure the file is available before running this line)
data = pd.read_csv("used_cars_data.csv")

# Import KaggleHub to download datasets from Kaggle
import kagglehub

# Download the latest version of the dataset from Kaggle
path = kagglehub.dataset_download("sukhmanibedi/cars4u")

# Print the path where the dataset is downloaded 

print("/used_cars_data.csv", path)


print(data.head())  # Show first 5 rows of the dataset
print(data.tail())  # Show last 5 rows of the dataset
data.info()  # Display column types and non-null counts
print(data.nunique())  # Count of unique values in each column
print(data.isnull().sum())  # Count of missing values in each column
print(data.describe().T)  # Summary statistics of numerical columns

# Show detailed statistics, including categorical columns
print(data.describe(include='all').T)

# Loop through each column to analyze its distribution
for col in data:
    print(col)  # Print column name
    
    # Check if the column is numeric before calculating skewness
    if data[col].dtype in ['int64', 'float64']: 
        print('Skew :', round(data[col].skew(), 2))
        
        # Create figure for histogram and boxplot
        plt.figure(figsize=(15, 4))
        
        # Histogram plot
        plt.subplot(1, 2, 1)
        data[col].hist(grid=False)
        plt.ylabel('count')
        
        # Boxplot for outlier detection
        plt.subplot(1, 2, 2)
        sns.boxplot(x=data[col])
        
        # Show plots
        plt.show()
