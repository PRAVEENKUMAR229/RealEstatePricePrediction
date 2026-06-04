import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load cleaned data
df = pd.read_csv(r"C:\Users\HP\Desktop\RealEstatePricePrediction\data\processed\featured_data.csv")

# Load the saved model
with open(r"C:\Users\HP\Desktop\RealEstatePricePrediction\models\best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load column names
with open(r"C:\Users\HP\Desktop\RealEstatePricePrediction\models\columns.json", "r") as f:
    columns = json.load(f)["data_columns"]

print("Data shape:", df.shape)
print("Total columns:", len(columns))

from sklearn.model_selection import train_test_split

x = df.drop(columns=["price"])
y = df["price"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
y_pred = model.predict(x_test)
print("First 5 actual prices:", y_test.values[:5])
print("First 5 predicted prices:", np.round(y_pred[:5], 2))

#calculating errors
mae= mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)
print("MAE:",round(mae,2))
print("MSE:",round(mse,2))
print("RMSE:",round(rmse,2))
print("R2:",round(r2,4))

#scatter plot 
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linewidth=2, linestyle="--")
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted Prices")
plt.show()

#Residual plots
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5, color="steelblue")
plt.axhline(y=0, color="red", linewidth=2, linestyle="--")
plt.xlabel("Predicted Price (Lakhs)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.show()

#feature importance
importance = pd.Series(np.abs(model.coef_), index=x.columns)
top_features = importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Most Important Features")
plt.xlabel("Coefficient Value (Absolute)")
plt.ylabel("Features")
plt.show()

#custom prediction
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(np.array(columns) == "location_" + location)[0]
    
    x = np.zeros(len(columns))
    x[columns.index("total_sqft")] = sqft
    x[columns.index("bath")] = bath
    x[columns.index("bhk")] = bhk
    x[columns.index("price_per_sqft")] = df["price_per_sqft"].mean()
    
    if len(loc_index) > 0:
        x[loc_index[0]] = 1
    
    return round(model.predict([x])[0], 2)

print(predict_price("Whitefield", 1000, 2, 2))
print(predict_price("Indiranagar", 1000, 2, 2))
print(predict_price("Cunningham Road", 1000, 2, 2))