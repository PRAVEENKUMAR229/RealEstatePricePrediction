import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\HP\Desktop\RealEstatePricePrediction\data\processed\cleaned_data.csv")
print(df.shape)
df.head()
print("Total unique locations:", df["location"].nunique())
print("\nLocation counts:\n", df["location"].value_counts().tail(20))

location_counts = df["location"].value_counts()
rare_locations = location_counts[location_counts <= 10].index
df["location"] = df["location"].apply(lambda x: "other" if x in rare_locations else x)

print("Unique locations after grouping:", df["location"].nunique())

#one hot encoding for location
df = pd.get_dummies(df, columns=["location"])
print("Shape after encoding:", df.shape)
print(df.head())

df = df.astype({col: int for col in df.select_dtypes(include='bool').columns})
print(df.head())
df.head(8917)

df.to_csv(r"C:\Users\HP\Desktop\RealEstatePricePrediction\data\processed\featured_data.csv", index=False)
print("Feature engineered data saved!")
print("Final shape:", df.shape)

if __name__ == "__main__":
    print("feature_engineering.py executed successfully!")