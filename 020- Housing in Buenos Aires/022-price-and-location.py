import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wqet_grader
from IPython.display import VimeoVideo
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)
wqet_grader.init("Project 2 Assessment")


### Predicting Price with Location

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

    # Split lat-lon column (Task 2.2.2)
    df[["lat", "lon"]]= df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns ="lat-lon", inplace=True)
    return df


#Task 2.2.1: Use your wrangle function to create a DataFrame frame1 from the CSV file data/buenos-aires-real-estate-1.csv.
frame1 = wrangle("data/buenos-aires-real-estate-1.csv")
print(frame1.info())
frame1.head()

#Task 2.2.2: Add to the wrangle function below so that, in the DataFrame it returns, the "lat-lon" column is replaced by separate "lat" and "lon" columns. Don't forget to also drop the "lat-lon" column. Be sure to rerun all the cells above before you continue.

#Task 2.2.3: Use you revised wrangle function create a DataFrames frame2 from the file data/buenos-aires-real-estate-2.csv.
frame2 = wrangle("data/buenos-aires-real-estate-2.csv")
print(frame2.info())
frame2.head()

#Task 2.2.4: Use pd.concat to concatenate frame1 and frame2 into a new DataFrame df. Make sure you set the ignore_index argument to True.
df = pd.concat([frame1, frame2], ignore_index=True)
print(df.info())
df.head()

# Explore

#Task 2.2.5: Complete the code below to create a Mapbox scatter plot that shows the location of the apartments in df.
fig = px.scatter_mapbox(
    df,  # Our DataFrame
    lat="lat",
    lon="lon",
    width=600,  # Width of map
    height=600,  # Height of map
    color="price_aprox_usd",
    hover_data=["price_aprox_usd"],  # Display price when hovering mouse over house
)

fig.update_layout(mapbox_style="open-street-map")

fig.show()

#Task 2.2.6: Complete the code below to create a 3D scatter plot, with "lon" on the x-axis, "lat" on the y-axis, and "price_aprox_usd" on the z-axis.
# Create 3D scatter plot
fig = px.scatter_3d(
    df,
    x="lon",
    y="lat",
    z="price_aprox_usd",
    labels={"lon": "longitude", "lat": "latitude", "price_aprox_usd": "price"},
    width=600,
    height=500,
)

# Refine formatting
fig.update_traces(
    marker={"size": 4, "line": {"width": 2, "color": "DarkSlateGrey"}},
    selector={"mode": "markers"},
)

# Display figure
fig.show()

# Split

#Task 2.2.7: Create the feature matrix named X_train. It should contain two features: ["lon", "lat"].
features = ["lon", "lat"]
X_train = df[features]
X_train.shape

#Task 2.2.8: Create the target vector named y_train, which you'll use to train your model. Your target should be "price_aprox_usd". Remember that, in most cases, your target vector should be one-dimensional.
target = "price_aprox_usd"
y_train = df[target]
y_train.shape

# Build model

# Baseline

#Task 2.2.9: Calculate the mean of your target vector y_train and assign it to the variable y_mean.
y_mean = y_train.mean()

#Task 2.2.10: Create a list named y_pred_baseline that contains the value of y_mean repeated so that it's the same length at y_train.
y_pred_baseline = [y_mean]*len(y_train)
y_pred_baseline[:5]

#Task 2.2.11: Calculate the baseline mean absolute error for your predictions in y_pred_baseline as compared to the true targets in y_train.
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))

# Mean apt price 134732.97
# Baseline MAE: 45422.75

# Iterate

#Task 2.2.12: Instantiate a SimpleImputer named imputer.
# This "fills" the missing values, NaN, etc.
imputer = SimpleImputer()

#Task 2.2.13: Fit your transformer imputer to the feature matrix X.
imputer.fit(X_train)

#Task 2.2.14: Use your imputer to transform the feature matrix X_train. Assign the transformed data to the variable XT_train.
XT_train = imputer.transform(X_train)
pd.DataFrame(XT_train, columns=X_train.columns).info()

#Task 2.2.15: Create a pipeline named model that contains a SimpleImputer transformer followed by a LinearRegression predictor.
model = make_pipeline(
    SimpleImputer(),
    LinearRegression()
)

#Task 2.2.16: Fit your model to the data, X_train and y_train.
model.fit(X_train, y_train)

# Evaluate

#Task 2.2.17: Using your model's predict method, create a list of predictions for the observations in your feature matrix X_train. Name this list y_pred_training.
y_pred_training = model.predict(X_train)

# Task 2.2.18: Calculate the training mean absolute error for your predictions in y_pred_training as compared to the true targets in y_train.
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))
# Training MAE: 42962.72

#Task 2.2.19: Run the code below to import your test data buenos-aires-test-features.csv into a DataFrame and generate a Series of predictions using your model. Then run the following cell to submit your predictions to the grader.
X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test))
y_pred_test.head()

# Communicative results

#Task 2.2.20: Extract the intercept and coefficients for your model.
intercept = model.named_steps["linearregression"].intercept_.round()
coefficients = model.named_steps["linearregression"].coef_.round()

#Task 2.2.21: Complete the code below and run the cell to print the equation that your model has determined for predicting apartment price based on latitude and longitude.
print(
    
    f"price = {intercept} + ({coefficients} * longitude) + ({coefficients} * latitude)"
)
#price = 38113587.0 + ([196709. 765467.] * longitude) + ([196709. 765467.] * latitude)

#Task 2.2.22: Complete the code below to create a 3D scatter plot, with "lon" on the x-axis, "lat" on the y-axis, and "price_aprox_usd" on the z-axis.
# Create 3D scatter plot
fig = px.scatter_3d(
    df,
    x="lon",
    y="lat",
    z="price_aprox_usd",
    labels={"lon": "longitude", "lat": "latitude", "price_aprox_usd": "price"},
    width=600,
    height=500,
)

# Create x and y coordinates for model representation
x_plane = np.linspace(df["lon"].min(), df["lon"].max(), 10)
y_plane = np.linspace(df["lat"].min(), df["lat"].max(), 10)
xx, yy = np.meshgrid(x_plane, y_plane)

# Use model to predict z coordinates
z_plane = model.predict(pd.DataFrame({"lon": x_plane, "lat": y_plane}))
zz = np.tile(z_plane, (10, 1))

# Add plane to figure
fig.add_trace(go.Surface(x=xx, y=yy, z=zz))

# Refine formatting
fig.update_traces(
    marker={"size": 4, "line": {"width": 2, "color": "DarkSlateGrey"}},
    selector={"mode": "markers"},
)

# Display figure
fig.show()
