#3.2. Linear Regression with Time Series Data
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import pytz
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#PREPARE DATA

#IMPORT
#Task 3.2.1: Complete to the create a client to connect to the MongoDB server, assign the "air-quality" database to db, and assign the "nairobi" connection to nairobi.
client = MongoClient(host="localhost", port=27017)
db = client["air-quality"]
nairobi = db["nairobi"]

#Task 3.2.2: Complete the wrangle function below so that the results from the database query are read into the DataFrame df. Be sure that the index of df is the "timestamp" from the results.
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    df = pd.DataFrame(results).set_index("timestamp")
    
    # Localize timezone (Task 3.2.4)
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers (Task 3.2.6)
    df = df[df["P2"]< 500]

    # Resample to 1H window, ffill missing values (Task 3.2.9)
    df = df["P2"].resample("1H").mean().fillna(method="ffill").to_frame().head()

    # Add lag feature (Task 3.2.10)
    df["P2.L1"] = df["P2"].shift(1)

    # Drop NaN rows
    df.dropna(inplace=True)
    return df

#Task 3.2.3: Use your wrangle function to read the data from the nairobi collection into the DataFrame df.
df = wrangle(nairobi)
df.head(10)

#Task 3.2.4: Add to your wrangle function so that the DatetimeIndex for df is localized to the correct timezone, "Africa/Nairobi". Don't forget to re-run all the cells above after you change the function.

#EXPLORE
#Task 3.2.5: Create a boxplot of the "P2" readings in df.
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(kind="box", vert=False, title="Distribution of PM2.5 Readings", ax=ax)

#Task 3.2.6: Add to your wrangle function so that all "P2" readings above 500 are dropped from the dataset. Don't forget to re-run all the cells above after you change the function.

#Task 3.2.7: Create a time series plot of the "P2" readings in df.
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].plot(xlabel="Time", ylabel="PM2.5", title="PM2.5", ax=ax)

#Task 3.2.8: Add to your wrangle function to resample df to provide the mean "P2" reading for each hour. Use a forward fill to impute any missing values. Don't forget to re-run all the cells above after you change the function.
df["P2"].resample("1H").mean().isnull().sum() # 102 missing values
df["P2"].resample("1H").mean().fillna(method="ffill") # ffill -> forward filling

#Task 3.2.9: Plot the rolling average of the "P2" readings in df. Use a window size of 168 (the number of hours in a week).
fig, ax = plt.subplots(figsize=(15, 6))
df["P2"].rolling(168).mean().plot(ax=ax, ylabel="PM2.5", title="Weekly Rolling Average");

#Task 3.2.10: Add to your wrangle function to create a column called "P2.L1" that contains the mean"P2" reading from the previous hour. Since this new feature will create NaN values in your DataFrame, be sure to also drop null rows from df.

#Task 3.2.11: Create a correlation matrix for df.
df.corr()

#Task 3.2.12: Create a scatter plot that shows PM 2.5 mean reading for each our as a function of the mean reading from the previous hour. In other words, "P2.L1" should be on the x-axis, and "P2" should be on the y-axis. Don't forget to label your axes!
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x=df["P2.L1"], y=df["P2"])
ax.plot([0, 120], [0, 120], linestyle="--", color="orange")
plt.xlabel("P2.L1")
plt.ylabel("P2")
plt.title("PM2.5 Autocorrelation")

#SPLIT

#Task 3.2.13: Split the DataFrame df into the feature matrix X and the target vector y. Your target is "P2".
target = "P2"
y = df[target]
X = df.drop(columns=target)
X.head()

#Task 3.2.14: Split X and y into training and test sets. The first 80% of the data should be in your training set. The remaining 20% should be in the test set.
cutoff = int(len(X) * 0.8)

X_train, y_train = X.iloc[:cutoff], y.iloc[:cutoff]
X_test, y_test = X.iloc[cutoff:], y.iloc[cutoff:]

