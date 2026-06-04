#custom prediction
import numpy as np
import pandas as pd
import pickle
import json

# Load the model
with open(r"C:\Users\HP\Desktop\RealEstatePricePrediction\models\best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the columns
with open(r"C:\Users\HP\Desktop\RealEstatePricePrediction\models\columns.json", "r") as f:
    columns = json.load(f)["data_columns"]

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