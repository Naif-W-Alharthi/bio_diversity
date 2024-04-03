import pandas as pd
import os
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


df8 = pd.read_csv('Biodiversity/' + "RED_LIST_28032024135604276.csv", header=0)

df8 = df8[["Country", "Year", "Value"]]

df9 = pd.read_csv('Biodiversity/' + "EXP_PM2_5_29032024190056738.csv", header=0)

print("")