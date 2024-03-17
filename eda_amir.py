import pandas as pd
import os
import scipy.stats as stats
import numpy as np

# # This code is done using a different methodology compared to my teammate Naif, where  he tries to have an automated
# # code for all the files, I have instead gone through each file individually.

df1 = pd.read_excel('Biodiversity/' + "Agricultural Land.xlsx", sheet_name='Data', header=0)
df1.drop(["Country Code", "Series Name", "Series Code"], axis=1, inplace=True)
# These columns are dropped because "Country Code" is redundant when we have the countries name, and "Series Name",
# "Series Code" repeat the same values over and over again. Plus by transforming these columns from Wide Format into
# Long Format, they are also made redundant.

df1 = df1.iloc[:-51]
# The last 5 rows did not contain data, but instead where the data came from and other rows did not represent countries,
# but instead represent groups of counties joined by different criteria.

df1 = pd.melt(df1, id_vars=['Country Name'], var_name='Year', value_name='Agricultural land (% of land area)')

df1['Year'] = df1['Year'].str.split('[', expand=True)[0].str.strip()
# Trimmed the year values so that they only display their year.
df1 = df1.sort_values(by=['Country Name', 'Year'])

print("")

df2 = pd.read_csv('Biodiversity/' + "deforestation-co2-trade-by-product.csv", header=0)
df2.drop(["Code", "Year"], axis=1, inplace=True)
# These columns are dropped because "Code" is redundant when we have the countries name, and "Year" repeat the same
# values over and over again.

df2 = pd.melt(df2, id_vars=['Entity'], var_name='Products', value_name='CO2 (in Tonnes)')
df2 = df2.sort_values(by=['Entity', 'Products'])

print("")

df3 = pd.read_excel('Biodiversity/' + "Forest Area.xlsx", sheet_name='Data', header=0)
df3.drop(["Country Code", "Series Name", "Series Code"], axis=1, inplace=True)
# These columns are dropped because "Country Code" is redundant when we have the countries name, and "Series Name",
# "Series Code" repeat the same values over and over again. Plus by transforming these columns from Wide Format into
# Long Format, they are also made redundant.

df1 = df1.iloc[:-51]
# The last 5 rows did not contain data, but instead where the data came from and other rows did not represent countries,
# but instead represent groups of counties joined by different criteria.

df3 = pd.melt(df3, id_vars=['Country Name'], var_name='Year', value_name='Forest area (% of land area)')

df3['Year'] = df3['Year'].str.split('[', expand=True)[0].str.strip()
# Trimmed the year values so that they only display their year.
df3 = df3.sort_values(by=['Country Name', 'Year'])

print("")

df4 = pd.read_csv('Biodiversity/' + "GHG emissions per kilogram produced.csv", header=0)
df4.drop(["Year"], axis=1, inplace=True)
# The column "Year" repeat the same value "2010" over and over again, and so is removed for being redundant.
df4.rename(columns={"Entity": "Country Name"}, inplace=True)
# renaming the columns in order to uniformize it with the rest of the files

print("")

df5 = pd.read_csv('Biodiversity/' + "global-living-planet-index.csv", header=0)
df5.rename(columns={"Entity": "Country Name"}, inplace=True)
# renaming the columns in order to uniformize it with the rest of the files

print("")

df6 = pd.read_excel('Biodiversity/' + "Tree Cover Loss.xlsx", sheet_name='Data', header=0)
df6.drop(["Country Code", "Series Name", "Series Code"], axis=1, inplace=True)
# These columns are dropped because "Country Code" is redundant when we have the countries name, and "Series Name",
# "Series Code" repeat the same values over and over again. Plus by transforming these columns from Wide Format into
# Long Format, they are also made redundant.

df1 = df1.iloc[:-51]
# The last 5 rows did not contain data, but instead where the data came from and other rows did not represent countries,
# but instead represent groups of counties joined by different criteria.

df6 = pd.melt(df6, id_vars=['Country Name'], var_name='Year', value_name='Tree Cover Loss (hectares)')

df6['Year'] = df6['Year'].str.split('[', expand=True)[0].str.strip()
# Trimmed the year values so that they only display their year.
df6 = df6.sort_values(by=['Country Name', 'Year'])

print("")

df7 = pd.read_csv('Biodiversity/' + "wheat-yields.csv", header=0)
df7 = df7.dropna(subset=["Code"])
# Removing all none countries from the
df7.drop(["Code"], axis=1, inplace=True)
# These columns are dropped because "Code" is redundant when we have the countries name

df7.rename(columns={"Entity": "Country Name", 'Wheat yield': 'Wheat Yield (tonnes/km2)'}, inplace=True)
# renaming the columns in order to uniformize it with the rest of the files


df_list = [df1, df2, df3, df4, df5, df6, df7]
df_names = ['2_Clean Agricultural Land.csv',
            '2_Clean deforestation-co2-trade-by-product.csv',
            '2_Clean Forest Area.csv',
            '2_Clean GHG emissions per kilogram produced.csv',
            '2_Clean global-living-planet-index.csv',
            '2_Clean Tree Cover Loss.csv',
            '2_Clean wheat-yields.csv']
if not os.path.exists("Clean Biodiversity"):
    os.makedirs("Clean Biodiversity")

# # Cleaning
i = 0
for df in df_list:
    df = df.apply(
        lambda x: x if x.name in ['Country Name', 'Entity', 'Products', 'Code'] else pd.to_numeric(x, errors='coerce'))
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

    nan_percentage = df.isna().mean() * 100
    columns_to_keep = nan_percentage[nan_percentage <= 80].index
    df = df[columns_to_keep]
    df = df.dropna()
    print(df.isnull().sum())
    df.to_csv(os.path.join("Clean Biodiversity", df_names[i]), index=False)
    i = i + 1

#
df1 = pd.read_csv(os.path.join("Clean Biodiversity", '2_Clean Agricultural Land.csv'))
df2 = pd.read_csv(os.path.join("Clean Biodiversity", '2_Clean deforestation-co2-trade-by-product.csv'))
df3 = pd.read_csv(os.path.join("Clean Biodiversity", '2_Clean Forest Area.csv'))
df4 = pd.read_csv(os.path.join("Clean Biodiversity", '2_Clean GHG emissions per kilogram produced.csv'))
df5 = pd.read_csv(os.path.join("Clean Biodiversity", '2_Clean global-living-planet-index.csv'))
df6 = pd.read_csv(os.path.join("Clean Biodiversity", '2_Clean Tree Cover Loss.csv'))
df7 = pd.read_csv(os.path.join("Clean Biodiversity", '2_Clean wheat-yields.csv'))
df_list = [df1, df2, df3, df4, df5, df6, df7]

# # Statistics
i = 0
for df in df_list:
    # print(df.describe())
    # print(df.shape)
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and not col in ["Year", "Upper CI", "Lower CI"]:
            # print(df[col].unique())
            # print(df.value_counts(subset = df[col]))

            # Sample size
            n = len(df[col])

            # Sample mean and standard deviation
            sample_mean = np.mean(df[col])
            sample_std = np.std(df[col], ddof=1)  # Use ddof=1 for sample standard deviation

            # Confidence level (e.g., 95% confidence level)
            confidence_level = 0.95

            # Calculate the critical value (z-score for large sample sizes)
            critical_value = stats.norm.ppf((1 + confidence_level) / 2)  # Two-tailed test

            # Calculate the margin of error
            margin_of_error = critical_value * (sample_std / np.sqrt(n))

            margin_of_error_percentage = (margin_of_error / sample_mean) * 100

            print(f"Margin of Error for Metric {df_names[i]}: ", margin_of_error_percentage.round(2))
            i += 1

print("")
