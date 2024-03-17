import pandas as pd
import os

# # This code is done using a different methodology compared to my teammate Naif, where  he tries to have an automated
# # code for all the files, I have instead gone through each file individually.

df1 = pd.read_excel('Biodiversity/' + "Agricultural Land.xlsx", sheet_name='Data', header=0)
df1.drop(["Country Code", "Series Name", "Series Code"], axis=1, inplace=True)
# These columns are dropped because "Country Code" is redundant when we have the countries name, and "Series Name",
# "Series Code" repeat the same values over and over again. Plus by transforming these columns from Wide Format into
# Long Format, they are also made redundant.

df1 = pd.melt(df1, id_vars=['Country Name'], var_name='Year', value_name='Agricultural land (% of land area)')

df1 = df1.iloc[:-5]
# The last 5 rows did not contain data, but instead where the data came from
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

df3 = df3.iloc[:-5]
# The last 5 rows did not contain data, but instead where the data came from

df3 = pd.melt(df3, id_vars=['Country Name'], var_name='Year', value_name='Forest area (% of land area)')

df3['Year'] = df3['Year'].str.split('[', expand=True)[0].str.strip()
# Trimmed the year values so that they only display their year.
df3 = df3.sort_values(by=['Country Name', 'Year'])

print("")

df4 = pd.read_csv('Biodiversity/' + "GHG emissions per kilogram produced.csv", header=0)
df4.drop(["Year"], axis=1, inplace=True)
# The column "Year" repeat the same value "2010" over and over again, and so is removed for being redundant.

print("")

df5 = pd.read_csv('Biodiversity/' + "global-living-planet-index.csv", header=0)

print("")

df6 = pd.read_excel('Biodiversity/' + "Tree Cover Loss.xlsx", sheet_name='Data', header=0)
df6.drop(["Country Code", "Series Name", "Series Code"], axis=1, inplace=True)
# These columns are dropped because "Country Code" is redundant when we have the countries name, and "Series Name",
# "Series Code" repeat the same values over and over again. Plus by transforming these columns from Wide Format into
# Long Format, they are also made redundant.

df6 = df6.iloc[:-5]
# The last 5 rows did not contain data, but instead where the data came from

df6 = pd.melt(df6, id_vars=['Country Name'], var_name='Year', value_name='Tree Cover Loss (hectares)')

df6['Year'] = df6['Year'].str.split('[', expand=True)[0].str.strip()
# Trimmed the year values so that they only display their year.
df6 = df6.sort_values(by=['Country Name', 'Year'])

print("")

df7 = pd.read_csv('Biodiversity/' + "wheat-yields.csv", header=0)
df7.drop(["Code"], axis=1, inplace=True)
# These columns are dropped because "Code" is redundant when we have the countries name

df_list = [df1,df2,df3,df4,df5,df6,df7]
df_names = ['2_Clean Agricultural Land.csv',
            '2_Clean deforestation-co2-trade-by-product.csv',
            '2_Clean Forest Area.csv',
            '2_Clean GHG emissions per kilogram produced.csv',
            '2_Clean global-living-planet-index.csv',
            '2_Clean Tree Cover Loss.csv',
            '2_Clean wheat-yields.csv']
if not os.path.exists("Clean Biodiversity"):
    os.makedirs("Clean Biodiversity")

df1.to_csv(os.path.join("Clean Biodiversity", '1_Long Agricultural Land.csv'), index=False)
df2.to_csv(os.path.join("Clean Biodiversity", '1_Long deforestation-co2-trade-by-product.csv'), index=False)
df3.to_csv(os.path.join("Clean Biodiversity", '1_Long Forest Area.csv'), index=False)
df4.to_csv(os.path.join("Clean Biodiversity", '1_Long GHG emissions per kilogram produced.csv'), index=False)
df5.to_csv(os.path.join("Clean Biodiversity", '1_Long global-living-planet-index.csv'), index=False)
df6.to_csv(os.path.join("Clean Biodiversity", '1_Long Tree Cover Loss.csv'), index=False)
df7.to_csv(os.path.join("Clean Biodiversity", '1_Long wheat-yields.csv'), index=False)

# # Cleaning
i=0
for df in df_list:
    df = df.apply(lambda x: x if x.name in ['Country Name', 'Entity', 'Products', 'Code'] else pd.to_numeric(x, errors='coerce'))
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

    nan_percentage = df.isna().mean() * 100
    columns_to_keep = nan_percentage[nan_percentage <= 80].index
    df = df[columns_to_keep]
    df = df.dropna()
    df.to_csv(os.path.join("Clean Biodiversity", df_names[i]), index=False)
    i = i + 1

# # 

