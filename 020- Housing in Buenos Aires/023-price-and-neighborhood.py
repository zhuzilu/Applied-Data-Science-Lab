### Predicting Price with Neighborhood
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wqet_grader
from category_encoders import OneHotEncoder
from IPython.display import VimeoVideo
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

    # Extract neighborhood
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)

    return df

#Task 2.3.1: Use glob to create a list that contains the filenames for all the Buenos Aires real estate CSV files in the data directory. Assign this list to the variable name files.
# GLOB -> import several csv files at the same time
files = glob("data/buenos-aires-real-estate-*.csv")


#Task 2.3.2: Use your wrangle function in a for loop to create a list named frames. The list should the cleaned DataFrames created from the CSV filenames your collected in files.
frames = []
for file in files:
    df = wrangle(file)
    print(df.shape)

#Task 2.3.3: Use pd.concat to concatenate the items in frames into a single DataFrame df. Make sure you set the ignore_index argument to True.
df = pd.concat(frames, ignore_index=True)
df.head()


# Explore
#Task 2.3.4: Modify your wrangle function to create a new feature "neighborhood". You can find the neighborhood for each property in the "place_with_parent_names" column. For example, a property with the place name "|Argentina|Capital Federal|Palermo|" is located in the neighborhood is "Palermo". Also, your function should drop the "place_with_parent_names" column.
#add to the wrangle function
# df["place_with_parent_names"].str.split("|", expand=True)[3].head()

# Split
#Task 2.3.5: Create your feature matrix X_train and target vector y_train. X_train should contain one feature: "neighborhood". Your target is "price_aprox_usd".
target = "price_aprox_usd"
features = ["neighborhood"]
y_train = df[target]
X_train = df[features]

# Build model

# Baseline
#Task 2.3.6: Calculate the baseline mean absolute error for your model.
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
print("Mean apt price:", y_mean)
# Mean apt price: 132383.83701458533
print("Baseline MAE:", mean_absolute_error(y_train, y_pred_baseline))
# Baseline MAE: 44860.10834274134

# Iterate
# Task 2.3.7: First, instantiate a OneHotEncoder named ohe. Make sure to set the use_cat_names argument to True. Next, fit your transformer to the feature matrix X_train. Finally, use your encoder to transform the feature matrix X_train, and assign the transformed data to the variable XT_train.
# Instantiate
ohe = OneHotEncoder(use_cat_names=True)
#Fit
ohe.fit(X_train)
#Transform
XT_train = ohe.transform(X_train)
print(XT_train.shape)
XT_train.head()

#Task 2.3.8: Create a pipeline named model that contains a OneHotEncoder transformer and a LinearRegression predictor. Then fit your model to the training data.
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LinearRegression()
)

model.fit(X_train, y_train)

# Evaluate
#Task 2.3.9: First, create a list of predictions for the observations in your feature matrix X_train. Name this list y_pred_training. Then calculate the training mean absolute error for your predictions in y_pred_training as compared to the true targets in y_train.
y_pred_training = model.predict(X_train)
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))
#Training MAE: 39342.97

#Task 2.3.10: Run the code below to import your test data buenos-aires-test-features.csv into a DataFrame and generate a Series of predictions using your model. Then run the following cell to submit your predictions to the grader. 
X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()


# Communicative results
#Task 2.3.11: Extract the intercept and coefficients for your model.
intercept = model.named_steps["linearregression"].intercept_
coefficients = model.named_steps["linearregression"].coef_
print("coefficients len:", len(coefficients)) #coefficients len: 57
print(coefficients[:5])  # First five coefficients [2.28511506e+17 2.28511506e+17 2.28511506e+17 2.28511506e+17 2.28511506e+17]


#Task 2.3.12: Extract the feature names of your encoded data from the OneHotEncoder in your model.
feature_names = model.named_steps["onehotencoder"].get_feature_names()
print("features len:", len(feature_names))
print(feature_names[:5])  # First five feature names

#Task 2.3.13: Create a pandas Series named feat_imp where the index is your features and the values are your coefficients.
feat_imp = pd.Series(coefficients, index=feature_names)
feat_imp.head()

#Task 2.3.14: Run the cell below to print the equation that your model has determined for predicting apartment price based on longitude and latitude.
print(f"price = {intercept.round(2)}")
for f, c in feat_imp.items():
    print(f"+ ({round(c, 2)} * {f})")

#Task 2.3.15: Scroll up, change the predictor in your model to Ridge, and retrain it. Then evaluate the model's training and test performance. Do you still have an overfitting problem? If not, extract the intercept and coefficients again (you'll need to change your code a little bit) and regenerate the model's equation. Does it look different than before?
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    Ridge()
)

model.fit(X_train, y_train)

#Task 2.3.16: Create a horizontal bar chart that shows the top 15 coefficients for your model, based on their absolute value.
feat_imp.sort_values(key=abs).tail(15).plot(kind="barh")
plt.xlabel("Importance [USD]")
plt.ylabel("Feature")
plt.title("Feature Importance for Apartment Price")

