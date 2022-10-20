#2.5. Predicting Apartment Prices in Mexico City

# Import libraries here
import warnings
import wqet_grader
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 2 Assessment")

#Task 2.5.1: Write a wrangle function that takes the name of a CSV file as input and returns a DataFrame. The function should do the following steps:
# Build your `wrangle` function
# Import
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Distrito Federal", less than 100,000
    mask_ba = df["place_with_parent_names"].str.contains("Distrito Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 100_000
    df = df[mask_ba & mask_apt & mask_price]
    
    # Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]
    
    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)
    
    # Create borough feature
    df["borough"] = df["place_with_parent_names"].str.split("|", expand=True)[1]
    df.drop(columns="place_with_parent_names", inplace=True)
    
    # Drop columns with more than 50% null values
    df.drop(columns=["floor", "expenses", "rooms"],inplace=True)
    
    # Drop low- and high-cardinality categorical variables
    df.drop(columns=["operation", "property_type", "currency", "properati_url"], inplace=True)
    
    # Drop leaky columns
    df.drop(columns=[
        'price',
        'price_aprox_local_currency',
        'price_per_m2',
        'price_usd_per_m2',
        ], 
        inplace=True)
    
    # Drop columns with multicollinearlity
    df.drop(columns=["surface_total_in_m2"], inplace=True)

    return df

# Task 2.5.2: Use glob to create the list files. It should contain the filenames of all the Mexico City real estate CSVs in the ./data directory, except for mexico-city-test-features.csv.
files = glob("data/mexico-city-real-estate-*.csv")

#Task 2.5.3: Combine your wrangle function, a list comprehension, and pd.concat to create a DataFrame df. It should contain all the properties from the five CSVs in files.
df= pd.concat([wrangle(file) for file in files], ignore_index=True)

#Task 2.5.4: Create a histogram showing the distribution of apartment prices ("price_aprox_usd") in df. Be sure to label the x-axis "Price [$]", the y-axis "Count", and give it the title "Distribution of Apartment Prices". Use Matplotlib (plt).
# Build histogram
plt.hist(df["price_aprox_usd"])


# Label axes
plt.xlabel("Price [$]")
plt.ylabel("Count")

# Add title
plt.title("Distribution of Apartment Prices");

# Don't delete the code below 👇
plt.savefig("images/2-5-4.png", dpi=150)

#Task 2.5.5: Create a scatter plot that shows apartment price ("price_aprox_usd") as a function of apartment size ("surface_covered_in_m2"). Be sure to label your axes "Price [USD]" and "Area [sq meters]", respectively. Your plot should have the title "Mexico City: Price vs. Area". Use Matplotlib (plt).
# Build scatter plot
plt.scatter(y=df["price_aprox_usd"], x=df["surface_covered_in_m2"])


# Label axes
plt.xlabel("Price [USD]")
plt.ylabel("Area [sq meters]")


# Add title
plt.title("Mexico City: Price vs. Area")

# Don't delete the code below 👇
plt.savefig("images/2-5-5.png", dpi=150)

#Task 2.5.7: Create your feature matrix X_train and target vector y_train. Your target is "price_aprox_usd". Your features should be all the columns that remain in the DataFrame you cleaned above.
# Split data into feature matrix `X_train` and target vector `y_train`.

features =["surface_covered_in_m2", "lat", "lon", "borough"]
target = "price_aprox_usd"
y_train = df[target]
X_train = df[features]

#Task 2.5.8: Calculate the baseline mean absolute error for your model.
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)

# Task 2.5.9: Create a pipeline named model that contains all the transformers necessary for this dataset and one of the predictors you've used during this project. Then fit your model to the training data.
# Build Model
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)
# Fit model
model.fit(X_train, y_train)

#Task 2.5.10: Read the CSV file mexico-city-test-features.csv into the DataFrame X_test.
X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())
X_test.head()

#Task 2.5.11: Use your model to generate a Series of predictions for X_test. When you submit your predictions to the grader, it will calculate the mean absolute error for your model.

y_test_pred =pd.Series(model.predict(X_test))
y_test_pred.head()