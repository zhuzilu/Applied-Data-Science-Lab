import matplotlib.pyplot as pyplot # Visualization
import pandas as pd # EDA
import plotly.express as px # Visualization

# Exploratory Data Analysis (EDA)

# 1st Step - Import Data

df = pd.read_csv("data/mexico-real-estate-clean.csv")

df.shape
df.head()
df.info()

# There are two Dtype data types: objects and float64
# But three categories of data: location, categorical and numeric.

# LOCATION DATA "lat" and "lon"

# The best way to visualize is to create a scatter plot on top of a map -> scatter_mapbox from the plotly library

fig = px.scatter_mapbox(
    df, # Our DataFrame
    lat = df["lat"],
    lon = df["lon"],
    center={"lat": 19.43, "lon": -99.13}, # To center the map on Mexico City
    width=600, 
    height=600, # width & height of the map
    hover_data=["price_usd"], # Display price when hovering mouse over house
)

fig.update_layout(mapbox_style="open-streep-map")
fig.show()
# CATEGORICAL DATA "state"

print(df["state"].nunique()) # It prints 30, because there are 30 unique values
print(df["state"].unique()) 

# It will print an array with all the unique values

array(['Estado de México', 'Nuevo León', 'Guerrero', 'Yucatán',
       'Querétaro', 'Morelos', 'Chiapas', 'Tabasco', 'Distrito Federal',
       'Nayarit', 'Puebla', 'Veracruz de Ignacio de la Llave', 'Sinaloa',
       'Tamaulipas', 'Jalisco', 'San Luis Potosí', 'Baja California',
       'Hidalgo', 'Quintana Roo', 'Sonora', 'Chihuahua',
       'Baja California Sur', 'Zacatecas', 'Aguascalientes', 'Guanajuato',
       'Durango', 'Tlaxcala', 'Colima', 'Oaxaca', 'Campeche'],
      dtype=object)

df["state"].value_counts() # Prints a series with name of state and count
df["state"].value_counts().head(10) # TOP TEN MOST COMMON STATES in our dataset


# NUMERIC DATA "area_m2" and "price_usd"
# .describe() shows the count, mean, std, min, 25%, 50%, 75%, max
df[["area_m2", "price_usd"]].describe()
#std standard deviation: spread of the data points
# 50% quartile also called the median


# Histogram of Area
plt.hist(df["area_m2"]);
plt.xlabel("Area [sq meters]")
plt.ylabel("Frequency")
plt.title("Distribution of Home Sizes")

# Very important to tag the labels of the x-axis and y-axis and title

# We can see a skewed distribution, a lot more houses that are smaller
# If we write a ; at the end we take off the array [] at the beginning


# Boxplot of area
plt.boxplot(df["area_m2"], vert=False)
plt.xlabel("Area [sq meters]")
plt.title("Distribution of Home Sizes");

# 50% of the data is located in the rectangle. Start -> 25%. Middle line -> 50%, final line -> 75%
# And the whiskers are the min and max

# Histogram of Price
plt.hist(df["price_usd"])
plt.xlabel("Price [USD]")
plt.ylabel("Frequency")
plt.title("Distribution of Home Prices");

# Boxplot of Price
plt.boxplot(df["price_usd"], vert=False)
plt.xlabel("Price [USD]")
plt.title("Ditribution of Home Prices");

# Outliers -> Value is more than 3 way deviation from your mean is represented as a circle

