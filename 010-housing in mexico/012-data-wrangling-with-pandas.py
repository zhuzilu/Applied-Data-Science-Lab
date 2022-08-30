import pandas as pd

# In this lessons we have 3 CSV files

#Task 1.2.1: Read these three files into three separate DataFrames named df1, df2, and df3, respectively.
df1 = pd.read_csv("/data/mexico-real-estate-1.csv")
df2 = pd.read_csv("/data/mexico-real-estate-2.csv")
df3 = pd.read_csv("/data/mexico-real-estate-3.csv")

# Clean

# .shape shows (observations, features) -> (700, 6)
print(df1.shape)

# .info -> nº entries, columns. Names of columns. Number of valid values and missing values. And data types for each columns. Memory space used.
print(df1.info())

# .head() shows the first 5 rows. If you add a number .head(2) or .head(20) it shows that number of rows
print(df1.head())

# Remove NaN not a number, blank cells missing values
df1.dropna(inplace=True)

# Price has a $ and , so to use it in operations we have to change it to a float.
# First we get the series (column)
df1["price_usd"]
# Then we use the .str.replace
# We also add the .astype(float) to convert it to a float
df1["price_usd"] = (
    df1["price_usd"]
    .str.replace("$", "", regex=False)
    .str.replace(",","")
    .astype(float)
)

# Cleaning df2
print(df2.shape)
print(df2.info())
print(df2.head(10))

df2.dropna(inplace=True)
print(df2.info())
print(df2.head(10))

# Create a "price_usd" column
df2["price_usd"] = (df2["price_mxn"] / 19).round(2)

# Drop the "price_mxn" column
df2.drop(columns=["price_mxn"], inplace=True)

# Cleaning df3
print(df3.shape)
print(df3.info())
print(df3.head(10))

df3.dropna(inplace=True)
print(df3.info())
print(df3.head(10))

#Instead of separate "lat" and "lon" columns, there's a single "lat-lon" column.
print(df3["lat-lon"].head())
df3[["lat", "lon"]] = df3["lat-lon"].str.split(",", expand=True)
# Split ("caracter que divide", expand=True (los pone en columnas diferentes))

# Instead of a "state" column, there's a "place_with_parent_names" column.
df3["state"]=df3["place_with_parent_names"].str.split("|", expand=True)[2].head()

# Drop columns
df3.drop(columns=["place_with_parent_names", "lat-lon"], inplace=True)

# CONCATENATE DATAFRAMES
#Two ways horizontal or vertical axis
#Horizontal is called axis = 1
# Vertical is called axis = 0, to remember think of gravity, gravity 0, it goes down.
# axis 0 is the default

df = pd.concat([df1, df2, df3])
print(df.shape)
df.head()

# Save df to csv, index=False so the index column doesn't appear
df.to_csv("data/mexico-real-estate-clean.csv", index=False)

