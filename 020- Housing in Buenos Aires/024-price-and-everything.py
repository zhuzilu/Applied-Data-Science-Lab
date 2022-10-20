### Predicting Price with Size, Location and neighborhood
import warnings
from glob import glob

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
#Prepare Data

# Import
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)

    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)

    # Drop features with high null counts

    df.drop(columns=["floor", "expenses"],inplace=True)
    
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
    df.drop(columns=["surface_total_in_m2", "rooms"], inplace=True)
    return df

#Task 2.4.1: Use glob to create a list that contains the filenames for all the Buenos Aires real estate CSV files in the data directory. Assign this list to the variable name files.
files = glob("data/buenos-aires-real-estate-*.csv")
files

#Task 2.4.2: Use your wrangle function in a list comprehension to create a list named frames. The list should contain the cleaned DataFrames for the filenames your collected in files.
frames = [wrangle(file) for file in files]

#Task 2.4.3: Use pd.concat to concatenate it items in frames into a single DataFrame df. Make sure you set the ignore_index argument to True.
df = pd.concat(frames, ignore_index=True)
print(df.info())
df.head()

# Last two tasks in one line of code:
# df = pd.concat([wrangle(file) for file in files], ignore_index=True)

# Explore
#Task 2.4.4: Modify your wrangle function to drop any columns that are more than half NaN values. Be sure to rerun all the cells above before you continue.

#Task 2.4.5: Calculate the number of unique values for each non-numeric feature in df.
df.select_dtypes("object").head()
df.select_dtypes("object").nunique() # Operation (sell), and apartment don't provide any information, because they are all the same.


#Task 2.4.6: Modify your wrangle function to drop high- and low-cardinality categorical features.

#Task 2.4.7: Modify your wrangle function to drop any features that would constitute leakage.

#Task 2.4.8: Plot a correlation heatmap of the remaining numerical features in df. Since "price_aprox_usd" will be your target, you don't need to include it in your heatmap.
corr = df.select_dtypes("number").drop(columns="price_aprox_usd").corr()
sns.heatmap(corr)

#Task 2.4.9: Modify your wrangle function to remove columns so that there are no strongly correlated features in your feature matrix.
target = "price_aprox_usd"
features = ["surface_covered_in_m2", "lat", "lon", "neighborhood"]
y_train = df[target]
X_train = df[features]

# Split
#Task 2.4.10: Create your feature matrix X_train and target vector y_train. Your target is "price_aprox_usd". Your features should be all the columns that remain in the DataFrame you cleaned above.
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
print("Mean apt price:", round(y_mean, 2))

print("Baseline MAE:", mean_absolute_error(y_train, y_pred_baseline))

# Mean apt price: 132383.84
# Baseline MAE: 44860.10834274134


# Build model

# Baseline
#Task 2.4.11: Calculate the baseline mean absolute error for your model.
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
print("Mean apt price:", round(y_mean, 2))

print("Baseline MAE:", mean_absolute_error(y_train, y_pred_baseline))

# Iterate
# Task 2.4.12: Create a pipeline named model that contains a OneHotEncoder, SimpleImputer, and Ridge predictor.

model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()
)


model.fit(X_train, y_train)

# Evaluate

#Task 2.4.13: Calculate the training mean absolute error for your predictions as compared to the true targets in y_train.
y_pred_training = model.predict(X_train)
print("Training MAE:", mean_absolute_error(y_train, y_pred_training))

#Task 2.4.14: Run the code below to import your test data buenos-aires-test-features.csv into a DataFrame and generate a list of predictions using your model. Then run the following cell to submit your predictions to the grader.
X_test = pd.read_csv("data/buenos-aires-test-features.csv")
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()


# Communicative results

#Task 2.4.15: Create a function make_prediction that takes four arguments (area, lat, lon, and neighborhood) and returns your model's prediction for an apartment price.
def make_prediction(area, lat, lon, neighborhood):
    data = {
        "surface_covered_in_m2": area,
        "lat": lat,
        "lon": lon,
        "neighborhood": neighborhood
    }
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df).round(2)[0]
    return f"Predicted apartment price: ${prediction}"

# Let's see if your function works. Run the cell below to find out!
make_prediction(110, -34.60, -58.46, "Villa Crespo")
# 'Predicted apartment price: $250775.11'

#Task 2.4.16: Add your make_prediction to the interact widget below, run the cell, and then adjust the widget to see how predicted apartment price changes.
interact(
    make_prediction,
    area=IntSlider(
        min=X_train["surface_covered_in_m2"].min(),
        max=X_train["surface_covered_in_m2"].max(),
        value=X_train["surface_covered_in_m2"].mean(),
    ),
    lat=FloatSlider(
        min=X_train["lat"].min(),
        max=X_train["lat"].max(),
        step=0.01,
        value=X_train["lat"].mean(),
    ),
    lon=FloatSlider(
        min=X_train["lon"].min(),
        max=X_train["lon"].max(),
        step=0.01,
        value=X_train["lon"].mean(),
    ),
    neighborhood=Dropdown(options=sorted(X_train["neighborhood"].unique())),
);