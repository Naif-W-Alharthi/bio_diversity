import pandas as pd
import os
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

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

df3 = df3.iloc[:-51]
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

df6 = df6.iloc[:-51]
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

print("")


df_list = [df1, df2, df3, df4, df5, df6, df7]
df_names = ['2_Clean Agricultural Land.csv',
            '2_Clean deforestation-co2-trade-by-product.csv',
            '2_Clean Forest Area.csv',
            '2_Clean GHG emissions per kilogram produced.csv',
            '2_Clean global-living-planet-index.csv',
            '2_Clean Tree Cover Loss.csv',
            '2_Clean wheat-yields.csv',
            '2_Clean Red List.csv']
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


# # Statistics/Viz


def remove_outlier(df_original, column_name, output_name):
    nrow1 = df_original.shape[0]
    df_original = df_original[df_original[column_name] > 0]
    Q1 = df_original[column_name].quantile(0.25)
    Q3 = df_original[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_original[(df_original[column_name] < lower_bound) | (df_original[column_name] > upper_bound)]
    df_original = df_original[~((df_original[column_name] < lower_bound) | (df_original[column_name] > upper_bound))]
    df_original.to_csv(os.path.join("Clean Biodiversity", output_name), index=False)
    print(output_name, " ", nrow1 - df_original.shape[0], " rows were removed.")
    return df_original


def plot_data(df_original, df_name):
    for column in df_original.columns:
        if df_original[column].dtype in ['int64', 'float64'] and column not in ["Year", "Upper CI", "Lower CI"]:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            ax1 = sns.histplot(data=df_original, x=df_original[column])
            ax1.set_xlabel(column, fontsize=15)
            ax1.set_ylabel('Count of records', fontsize=15)
            ax1.set_title(f'Univariate analysis of {df_name}', fontsize=20)
            plt.subplot(1, 2, 2)
            ax2 = sns.boxplot(data=df_original, x=df_original[column])
            ax2.set_title(f'Boxplot analysis of {df_name}', fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join("Clean Biodiversity", df_name + ".png"))


def join_all_df(Dfs, col_names):
    merged_df = Dfs[0][["Country Name", "Year"]]
    for ind, Df in enumerate(Dfs):
        df_merge = Df[["Country Name", "Year", col_names[ind]]]
        merged_df = pd.merge(merged_df, df_merge, on=["Country Name", "Year"], how='outer')
    col_names = ["Country Name", "Year"] + col_names
    merged_df.columns = col_names
    return merged_df


# ## deforestation-co2-trade-by-product.csv
df1 = remove_outlier(df1, "Agricultural land (% of land area)", "3_Outlier Agricultural Land.csv")
df2 = remove_outlier(df2, "CO2 (in Tonnes)", "3_Outlier deforestation-co2-trade-by-product.csv")
df3 = remove_outlier(df3, "Forest area (% of land area)", "3_Outlier Forest Area.csv")
df4 = remove_outlier(df4, "GHG emissions per kilogram (Poore & Nemecek, 2018)",
                     "3_Outlier GHG emissions per kilogram produced.csv")
df5 = remove_outlier(df5, "Living Planet Index", "3_Outlier global-living-planet-index.csv")
df6 = remove_outlier(df6, "Tree Cover Loss (hectares)", "3_Outlier Tree Cover Loss.csv")
df7 = remove_outlier(df7, "Wheat Yield (tonnes/km2)", "3_Outlier wheat-yields.csv")

i = 0
df_list = [df1, df2, df3, df4, df5, df6, df7]
for df in df_list:
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64'] and not col in ["Year", "Upper CI", "Lower CI"]:
            # Sample size/mean/standard deviations
            n = len(df[col])
            sample_mean = np.mean(df[col])
            sample_std = np.std(df[col], ddof=1)
            # Delta Degrees of Freedom = 1 for sample standard deviation

            confidence_level = 0.95
            # Confidence level (we will use a confidence level of 95% as not only is that one of the recommended/
            # standard levels used when calculating error margins, it is also the level used to determine the error
            # margins in of our files already (namely "global-living-planet-index").

            critical_value = stats.norm.ppf((1 + confidence_level) / 2)  # Two-tailed test
            # Calculating the critical value (z-score for large sample sizes)

            margin_of_error = critical_value * (sample_std / np.sqrt(n))
            # Calculating the margin of error
            print(f"Margin of Error for Metric {df_names[i]}: {sample_mean.round(2)} +- {margin_of_error.round(3)} ")
            margin_of_error_percentage = (margin_of_error / sample_mean) * 100

            print(f"Margin of Error for Metric {df_names[i]}: {margin_of_error_percentage.round(2)} %")
            i += 1

print("")

# plot_data(df1, "Agricultural Land")
# plot_data(df2, "Deforestation CO2 Trade by Product")
# plot_data(df3, "Forest Area")
# plot_data(df4, "GHG Emissions per Kilogram Produced")
# plot_data(df5, "Global Living Planet Index")
# plot_data(df6, "Tree Cover Loss")
# plot_data(df7, "Wheat Yields")

filtered_df = join_all_df([df1, df3, df6, df7], ["Agricultural land (% of land area)",
                                                 "Forest area (% of land area)",
                                                 "Tree Cover Loss (hectares)",
                                                 "Wheat Yield (tonnes/km2)"])

# filtered_df = filtered_df.dropna(how='any')
filtered_df.to_csv(os.path.join("Clean Biodiversity", 'Merged_data.csv'), index=False)

# grouped = filtered_df.groupby(["Country Name"])

# for country, group_df in grouped:
#     plt.figure(figsize=(12, 6))
#     for i, col in enumerate(group_df.columns[2:], start=1):
#         plt.subplot(1, len(group_df.columns[2:]), i)
#         sns.lineplot(x='Year', y=col, data=group_df)
#         plt.title(col)
#         plt.xlabel('Year')
#         plt.ylabel('Value')
#         plt.grid(True)
#     plt.suptitle(f'Changes over time for {country}')
#     plt.tight_layout()
#     plt.savefig(os.path.join("Clean Biodiversity", "Change_Over_Time", f'Changes over time for {country}.png'))


# Biodiversity Score
biodiversity_cols = ["Agricultural land (% of land area)",
                     "Forest area (% of land area)",
                     "Wheat Yield (tonnes/km2)"]

# ########### New regression ############# #
result_df = filtered_df.groupby('Year').agg({'Agricultural land (% of land area)': lambda x: x.mean(skipna=True),
                                             'Forest area (% of land area)': lambda x: x.mean(skipna=True),
                                             'Wheat Yield (tonnes/km2)': lambda x: x.mean(skipna=True)
                                             })

result_df = pd.merge(result_df, df5[df5['Country Name'] == 'World'][['Year', 'Living Planet Index']],
                     on='Year', how='left')

result_df = result_df.dropna()
# result_df = result_df[(result_df['Year'] >= 2002) & (result_df['Year'] <= 2018)]

result_df.to_csv(os.path.join("Clean Biodiversity", 'Regression_data.csv'), index=False)

# Select the specified columns
selected_columns = ["Agricultural land (% of land area)",
                    "Forest area (% of land area)",
                    "Wheat Yield (tonnes/km2)",
                    "Living Planet Index"]

# Create a DataFrame containing only the selected columns
selected_df = result_df[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Plot')
plt.show()
plt.savefig(os.path.join("Clean Biodiversity", 'Correlation Plot.png'))

X = result_df[biodiversity_cols]
X = sm.add_constant(X)
Y = result_df['Living Planet Index']

lmvr = sm.OLS(Y, X)  # X.astype(float))
lmvr_res = lmvr.fit()
print(lmvr_res.summary())

# Assuming df is your DataFrame containing the relevant columns

# Step 1: Normalize the feature columns
scaler = StandardScaler()
normalized_features = scaler.fit_transform(result_df[["Agricultural land (% of land area)",
                                                      "Forest area (% of land area)",
                                                      # "Tree Cover Loss (hectares)",
                                                      "Wheat Yield (tonnes/km2)"]])

# Step 2: Prepare the feature matrix (X) and target vector (y)
X = pd.DataFrame(normalized_features, columns=["Agricultural land (% of land area)",
                                               "Forest area (% of land area)",
                                               # "Tree Cover Loss (hectares)",
                                               "Wheat Yield (tonnes/km2)"])
Y = result_df['Living Planet Index']
X.reset_index(drop=True, inplace=True)
Y.reset_index(drop=True, inplace=True)
# Add constant to the features for intercept term
X = sm.add_constant(X)

# Step 3: Fit the OLS regression model
model = sm.OLS(Y, X).fit()

# Step 4: Interpret the results
print(model.summary())

# ########### New regression ############# #


# # Normalize the columns (useful for combining different scales)
# normalized_df = filtered_df[biodiversity_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# # Calculate the average biodiversity score
# normalized_df['Biodiversity Score'] = normalized_df.mean(axis=1)
# # Merge the biodiversity score with the original DataFrame
# result_df = pd.concat([filtered_df, normalized_df['Biodiversity Score']], axis=1)

# Coefficients


# X = result_df[biodiversity_cols]
# X = sm.add_constant(X)
# Y = result_df['Biodiversity Score']
#
# lmvr = sm.OLS(Y, X)  # X.astype(float))
# lmvr_res = lmvr.fit()
# print(lmvr_res.summary())
#
# # Fit linear regression model
# model = LinearRegression()
# model.fit(X, Y)
#
# # Get coefficients
# coefficients = model.coef_
#
# # Display coefficients for each column
# for i, col in enumerate(biodiversity_cols):
#     print(f"Coefficient for {col}: {coefficients[i]}")
#
# pred_ols = lmvr_res.get_prediction()
# iv_l = pred_ols.summary_frame()["obs_ci_lower"]
# iv_u = pred_ols.summary_frame()["obs_ci_upper"]
#
# num_cols = X.shape[1]
# fig, axes = plt.subplots(num_cols, 1, figsize=(8, 6 * num_cols))
# for i in range(num_cols):
#     axes[i].plot(X[:, i], Y, "o", label="Actual Biodiversity Score")
#     axes[i].plot(X[:, i], lmvr_res.fittedvalues, "r--.", label="Fitted Values")
#     axes[i].plot(X[:, i], iv_u, "g--", label="Upper Bound")
#     axes[i].plot(X[:, i], iv_l, "b--", label="Lower Bound")
#     axes[i].set_xlabel(f"X{i + 1}")
#     axes[i].set_ylabel("Y Label")
#     axes[i].legend(loc="best")
#
# plt.tight_layout()
# plt.savefig(os.path.join("Clean Biodiversity", 'Regression Results.png'))

# print(df.describe())
# print(df.shape)
# print(df[col].unique())
# print(df.value_counts(subset = df[col]))
# i = 0
# for df in df_list:
#     for col in df.columns:
#         if df[col].dtype in ['int64', 'float64'] and not col in ["Year", "Upper CI", "Lower CI"]:
#             plt.figure(figsize=(10, 5))
#             ax = sns.histplot(data=df, x=df[col])
#             ax.set_xlabel(col, fontsize=15)
#             ax.set_ylabel('Count of records', fontsize=15)
#             ax.set_title(f'Univariate analysis of {df_names[i]}', fontsize=20)
#
#             plt.figure(figsize=(10, 5))
#             ax = sns.boxplot(data=df, x=df[col])
#             ax.set_title(f'Boxplot analysis of {df_names[i]}', fontsize=20)
#
#             i += 1
