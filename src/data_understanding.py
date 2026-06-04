import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

df = pd.read_csv(r"C:\Users\HP\Desktop\RealEstatePricePrediction\data\raw\Bengaluru_House_Data.csv")
df.head()
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
df.describe()
print("Unique locations:", df["location"].nunique())
print("Unique area types:", df["area_type"].nunique())
print("Unique sizes:", df["size"].nunique())
print("\nSize values:\n", df["size"].value_counts().head(10))

#Histoplot
plt.figure(figsize=(12, 6))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price (Lakhs)")
plt.ylabel("Count")
plt.show()

#correlation heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df[["bath", "balcony", "price"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#top 10 locations
top_locations = df.groupby("location")["price"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_locations.values, y=top_locations.index)
plt.title("Top 10 Most Expensive Locations")
plt.xlabel("Average Price (Lakhs)")
plt.show()

if __name__ == "__main__":
    print("data_understanding.py executed successfully!")