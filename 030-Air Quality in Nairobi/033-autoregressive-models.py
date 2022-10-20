#3.3. Autoregressive Models
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from IPython.display import VimeoVideo
from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg

warnings.simplefilter(action="ignore", category=FutureWarning)

#PREPARE DATA
#IMPORT

#Task 3.3.1: Complete to the create a client to connect to the MongoDB server, assigns the "air-quality" database to db, and assigned the "nairobi" connection to nairobi.
client = MongoClient(host="localhost", port=27017)
db = client["air-quality"]
nairobi = db["nairobi"]

#Task 3.3.2: Change the wrangle function below so that it returns a Series of the resampled data instead of a DataFrame.
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 29, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )

    # Read data into DataFrame
    df = pd.DataFrame(list(results)).set_index("timestamp")

    # Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Nairobi")

    # Remove outliers
    df = df[df["P2"] < 500]

    # Resample to 1hr window
    y = df["P2"].resample("1H").mean().fillna(method='ffill').to_frame()

    return y

#Task 3.3.3: Use your wrangle function to read the data from the nairobi collection into the Series y.
y = wrangle(nairobi)
y.head()


#EXPLORE

#SPLIT

#CONNECT

#EXPLORE

#IMPORT

