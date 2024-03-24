import pandas as pd
import os
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

df1 = pd.read_excel('Biodiversity/' + "Agricultural Land.xlsx", sheet_name='Data', header=0)
df1.drop(["Country Code", "Series Name", "Series Code"], axis=1, inplace=True)
df1 = df1.iloc[[238]]
df1 = pd.melt(df1, id_vars=['Country Name'], var_name='Year', value_name='Agricultural land (% of land area)')
df1['Year'] = df1['Year'].str.split('[', expand=True)[0].str.strip()

print("")

df3 = pd.read_excel('Biodiversity/' + "Forest Area.xlsx", sheet_name='Data', header=0)
df3.drop(["Country Code", "Series Name", "Series Code"], axis=1, inplace=True)
df3 = df3.iloc[[238]]
df3 = pd.melt(df3, id_vars=['Country Name'], var_name='Year', value_name='Forest area (% of land area)')
df3['Year'] = df3['Year'].str.split('[', expand=True)[0].str.strip()

print("")

df5 = pd.read_csv('Biodiversity/' + "global-living-planet-index.csv", header=0)
df5.rename(columns={"Entity": "Country Name"}, inplace=True)
df5.drop(["Code", "Upper CI", "Lower CI"], axis=1, inplace=True)
df5 = df5.iloc[293:341]
df5["Year"] = df5["Year"].astype(str)

print("")

df6 = pd.read_excel('Biodiversity/' + "Tree Cover Loss.xlsx", sheet_name='Data', header=0)
df6.drop(["Country Code", "Series Name", "Series Code"], axis=1, inplace=True)
df6 = df6.iloc[[238]]
df6 = pd.melt(df6, id_vars=['Country Name'], var_name='Year', value_name='Tree Cover Loss (hectares)')
df6['Year'] = df6['Year'].str.split('[', expand=True)[0].str.strip()

print("")

df7 = pd.read_csv('Biodiversity/' + "wheat-yields.csv", header=0)
df7 = df7.dropna(subset=["Code"])
df7.drop(["Code"], axis=1, inplace=True)
df7 = df7.reset_index(drop=True)
df7 = df7.iloc[6530:6591]
df7.rename(columns={"Entity": "Country Name", 'Wheat yield': 'Wheat Yield (tonnes/km2)'}, inplace=True)
df7["Year"] = df7["Year"].astype(str)
# df1.to_csv(os.path.join("Clean Biodiversity", '4_World Agricultural Land.csv'), index=False)
# df3.to_csv(os.path.join("Clean Biodiversity", '4_World Forest Area.csv'), index=False)
# df5.to_csv(os.path.join("Clean Biodiversity", '4_World global-living-planet-index.csv'), index=False)
# df6.to_csv(os.path.join("Clean Biodiversity", '4_World Tree Cover Loss.csv'), index=False)
# df7.to_csv(os.path.join("Clean Biodiversity", '4_World wheat-yields.csv'), index=False)

# Merge the DataFrames on country_name and year
merged_df = pd.merge(df1, df3, on=['Country Name', 'Year'], how='outer')
# merged_df = pd.merge(merged_df, df6, on=['Country Name', 'Year'], how='outer')
merged_df = pd.merge(merged_df, df7, on=['Country Name', 'Year'], how='outer')
merged_df = pd.merge(merged_df, df5, on=['Country Name', 'Year'], how='outer')

merged_df.to_csv(os.path.join("Clean Biodiversity", '4_World All_metrics.csv'), index=False)
merged_df["Year"] = merged_df["Year"].astype(int)

merged_df = merged_df[(merged_df['Year'] >= 1992) & (merged_df['Year'] <= 2017)]

print(merged_df.head())

regression_columns = ['Agricultural land (% of land area)', 'Forest area (% of land area)', 'Wheat Yield (tonnes/km2)']
merged_df[regression_columns] = merged_df[regression_columns].astype(float)
merged_df['Living Planet Index'] = merged_df['Living Planet Index'].astype(float)

X = merged_df[regression_columns]
X = sm.add_constant(X)
Y = merged_df['Living Planet Index']

lmvr = sm.OLS(Y, X)  # X.astype(float))
lmvr_res = lmvr.fit()
print(lmvr_res.summary())
