# Tabular data 
# Table organised in cells with rows and columns. (Excel spreadsheet)

# Tidy data : 
# rows = observations
# columns = features
# cells = values


# List -> price in USD, area in m2, rooms
house_0_list = [115910.26, 128, 4]
print(house_0_list)

#Task 1.1.1: Can you use the information in this list to calculate the price per square meter for house_0?
price = house_0_list[0]
sqm = house_0_list[1]
house_0_price_m2 = price / sqm
print(house_0_price_m2)

#Task 1.1.2: Next, use the append method to add the price per square meter to the end of the end of house_0.
house_0_list.append(house_0_price_m2)

# Nested List: one list for each observation 
houses_nested_list = [
    [115910.26, 128.0, 4.0],
    [48718.17, 210.0, 3.0],
    [28977.56, 58.0, 2.0],
    [36932.27, 79.0, 3.0],
    [83903.51, 111.0, 3.0],
]

print(houses_nested_list)

# Task 1.1.3: Append the price per square meter to each observation in houses_nested_list using a for loop.
for house in houses_nested_list:
    price_m2 = house[0]/house[1]
    house.append(price_m2)

print(houses_nested_list)

# Dictionaries: each value is associated with a key

house_0_dict = {
    "price_aprox_usd": 115910.26,
    "surface_covered_in_m2": 128,
    "rooms": 4,
}

print(house_0_dict)

#Task 1.1.4: Calculate the price per square meter for house_0 and add it to the dictionary under the key "price_per_m2".

house_0_dict["price_per_m2"] = house_0_dict["price_aprox_usd"]/house_0_dict["surface_covered_in_m2"]
print(house_0_dict)

# To combine all the houses we can create a list of dictionaries

houses_rowwise = [
    {
        "price_aprox_usd": 115910.26,
        "surface_covered_in_m2": 128,
        "rooms": 4,
    },
    {
        "price_aprox_usd": 48718.17,
        "surface_covered_in_m2": 210,
        "rooms": 3,
    },
    {
        "price_aprox_usd": 28977.56,
        "surface_covered_in_m2": 58,
        "rooms": 2,
    },
    {
        "price_aprox_usd": 36932.27,
        "surface_covered_in_m2": 79,
        "rooms": 3,
    },
    {
        "price_aprox_usd": 83903.51,
        "surface_covered_in_m2": 111,
        "rooms": 3,
    },
]

print(houses_rowwise)

# This way of storing data -> JSON

#Task 1.1.5: Using a for loop, calculate the price per square meter and store the result under a "price_per_m2" key for each observation in houses_rowwise.
for house in houses_rowwise:
    house["price_per_m2"] = house["price_aprox_usd"]/house["surface_covered_in_m2"]
print(houses_rowwise)

#Task 1.1.6: To calculate the mean price for houses_rowwise by completing the code below.
house_prices = []
for house in houses_rowwise:
    house_prices.append(house["price_aprox_usd"])

mean_house_price = sum(house_prices) / len(house_prices)

print(mean_house_price)

# Data organized by features

houses_columnwise = {
    "price_aprox_usd": [115910.26, 48718.17, 28977.56, 36932.27, 83903.51],
    "surface_covered_in_m2": [128.0, 210.0, 58.0, 79.0, 111.0],
    "rooms": [4.0, 3.0, 2.0, 3.0, 3.0],
}

print(houses_columnwise)

# Task 1.1.7: Calculate the mean house price in houses_columnwise

mean_house_price = sum(houses_columnwise["price_aprox_usd"]) / len(houses_columnwise["price_aprox_usd"])

print(mean_house_price)

#Task 1.1.8: Create a "price_per_m2" column in houses_columnwise
price = houses_columnwise["price_aprox_usd"]
area = houses_columnwise["surface_covered_in_m2"]

price_per_m2 = []
for p, a in zip(price, area):
    price_m2 = p/a
    price_per_m2.append(price_m2)

houses_columnwise["price_per_m2"] = price_per_m2

print(houses_columnwise)

# Tabular Data and pandas DataFrames

# Let's import pandas and then create a DataFrame from houses_columnwise

import pandas as pd

data = {
    "price_aprox_usd": [115910.26, 48718.17, 28977.56, 36932.27, 83903.51],
    "surface_covered_in_m2": [128.0, 210.0, 58.0, 79.0, 111.0],
    "rooms": [4.0, 3.0, 2.0, 3.0, 3.0],
}

df_houses = pd.DataFrame(data)

print(df_houses)