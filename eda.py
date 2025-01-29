import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv("used_cars_data.csv")
print("top 5 row")
print(data.head())
print("last 5 row")
print(data.tail())
print("info")
print(data.info())
print(data.nunique())
print(data.isnull().sum())
print((data.isnull().sum()/(len(data)))*100)
data = data.drop(['S.No.'], axis = 1)
print(data.info())
from datetime import date
date.today().year
data['Car_Age']=date.today().year-data['Year']
print(data.head())
