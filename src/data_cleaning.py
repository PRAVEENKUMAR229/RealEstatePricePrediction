import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
df = pd.read_csv(r"C:\Users\HP\Desktop\RealEstatePricePrediction\data\raw\Bengaluru_House_Data.csv")
print(df.shape)
df.head()
df.drop(columns = ["society"], inplace = True)
print(df.columns.tolist())

# dropping the rows with missing values
df.dropna(inplace = True)
print("shape after dropping missing values:", df.shape)
df["bhk"] = df["size"].apply(lambda x: int(x.split(" ")[0]))
print(df["bhk"].value_counts().head(10))

# drop original size column
df.drop(columns = ["size"], inplace = True)
print(df.columns.tolist())

#total_sqft column
df[~df["total_sqft"].apply(lambda x: str(x).replace(".", "").isnumeric())]["total_sqft"].head(20)

def convert_sqft(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2 
    try:
        return float(x)
    except: 
        return None
df["total_sqft"] = df["total_sqft"].apply(convert_sqft)
df = df.dropna(subset = ["total_sqft"])
print(df.shape)
print(df["total_sqft"].head(10))

# removing outliers
# considering that a single bedroom with atleast 300sqft
# example: 2BHK with 400sqft then 400/2 = 200sqft per bedroom- remove
# 2BHK with 800sqft then 800/2 = 400sqft per bedroom- keep 
df = df[df["total_sqft"] / df["bhk"] >= 300]
print("Shape:", df.shape)

#adding price per sqft column
df["price_per_sqft"] = round(df["price"]*100000 / df["total_sqft"], 2)
print(df["price_per_sqft"].describe())

# removing price per sqft outliers per location
def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf["price_per_sqft"])
        st = np.std(subdf["price_per_sqft"])
        reduced_df = subdf[(subdf["price_per_sqft"] > (m - st)) & (subdf["price_per_sqft"] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df = remove_outliers(df)
print("Shape:", df.shape)

# removing bathroom outliers
print(df[df["bath"] > df["bhk"] + 2]["bath"].value_counts())
df = df[df["bath"] <= df["bhk"] + 2]
print(df.shape)

# droping unnecessary columns
df.drop(columns = ["availability","area_type","balcony"], inplace = True)
print(df.columns.tolist())

df.to_csv(r"C:\Users\HP\Desktop\RealEstatePricePrediction\data\processed\cleaned_data.csv", index=False)
print("Cleaned data saved successfully!")
print("Final shape:", df.shape)

if __name__ == "__main__":
    print("data_cleaning.py executed successfully!")