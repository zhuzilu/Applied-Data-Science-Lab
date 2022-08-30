# Location or Size: What Influences House Prices in Mexico?
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data/mexico-real-estate-clean.csv")

# Group by state

mean_price_by_state = df.groupby("state")["price_usd"].mean().sort_values(ascending=False)
mean_price_by_state

# Table grouped states and their mean price in USD

# Bar Chart (.plot using pandas, so the xlabel's syntax is different)

mean_price_by_state.plot(
    kind="bar",
    xlabel="State",
    ylabel="Price [USD]",
    title="Mean House Price by State"    
);

# Price per sq meter

(
    df.groupby("state")
    ["price_per_m2"].mean()
    .sort_values(ascending=False)
    .plot(
        kind="bar",
        xlabel = "State",
        ylabel = "Mean price per M^2[USD]", 
        title = "Mean House Price per M^2 by State"
    )
);


# Research question 2: Does the size of the house influence price?
plt.scatter(x=df["area_m2"], y=df["price_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Price vs Area");

# Correlation numerical using Correlation Coefficient
p_correlation = df["area_m2"].corr(df["price_usd"])
print(p_correlation)

#The correlation coefficient is over 0.5, so there's a moderate relationship house size and price in Mexico. But does this relationship hold true in every state? Let's look at a couple of states, starting with Morelos.


# Make a subset on one district
df_morelos = df[df["state"] == "Morelos"]

# Scatter plot in Morelos Price vs area, really strong
plt.scatter(x=df_morelos["area_m2"], y=df_morelos["price_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [usd]")
plt.title("Morelos: Price vs Area")

p_correlation = df_morelos["area_m2"].corr(df["price_usd"])
print(p_correlation) #strong correlation 0.849

# Subset `df` to include only observations from `"Distrito Federal"`
df_mexico_city = df[df["state"] == "Distrito Federal"]
df_mexico_city.head()

# Create a scatter plot price vs area
plt.scatter(x=df_mexico_city["area_m2"], y=df_mexico_city["price_usd"])
plt.xlabel("Area [sq meters]")
plt.ylabel("Price [USD]")
plt.title("Mexico DF price vs area")
p_correlation = df_mexico_city["area_m2"].corr(df_mexico_city["price_usd"])
print(p_correlation) #weak correlation 0.410

