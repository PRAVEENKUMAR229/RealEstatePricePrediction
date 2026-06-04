import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import json
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12,6)

df = pd.read_csv(r"C:\Users\HP\Desktop\RealEstatePricePrediction\data\processed\featured_data.csv")

print(df.shape)
df.head()

x = df.drop(columns = ["price"]) #all columns except price are features
y = df["price"]
print(x.shape)
print(y.shape)
print(y.head().values)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42) #80% training set, 20% test set
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Linear Regression model
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
print("Training Score:", lr_model.score(x_train, y_train)) # R squared
print("Test Score:", lr_model.score(x_test, y_test))

#cross validation
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
scores = cross_val_score(lr_model, x, y, cv=cv)
print("Cross Validation Scores:", np.round(scores, 4))
print("Mean Score:", round(scores.mean(), 4))
print("Std Deviation:", round(scores.std(), 4))

#Decision Tree model
dt_model = DecisionTreeRegressor(random_state = 42)
dt_model.fit(x_train,y_train)
print("Training Score:", (dt_model.score(x_train,y_train)))
print("Test Score:", (dt_model.score(x_test,y_test)))
dt_scores = cross_val_score(dt_model,x,y, cv = cv)
print("CV Mean Score:", dt_scores.mean())
print("CV standard deviation:", dt_scores.std())

#Lasso model
ls_model = Lasso(alpha = 1.0)
ls_model.fit(x_train,y_train)
print("Training Score:", (ls_model.score(x_train,y_train)))
print("Test Score:", (ls_model.score(x_test,y_test)))
ls_scores = cross_val_score(ls_model,x,y, cv = cv)
print("CV Mean Score:", ls_scores.mean())
print("CV standard deviation:", ls_scores.std())

import pickle
import json

# Save the model
with open(r"C:\Users\HP\Desktop\RealEstatePricePrediction\models\best_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

# Save the column names
columns = x.columns.tolist()
with open(r"C:\Users\HP\Desktop\RealEstatePricePrediction\models\columns.json", "w") as f:
    json.dump({"data_columns": columns}, f)

print("Model saved successfully!")
print("Columns saved successfully!")

#model comparison chart
models = ["Linear Regression", "Decision Tree", "Lasso"]
train_scores = [0.8654, 1.0, 0.8450]
test_scores = [0.9216, 0.7418, 0.9081]
cv_scores = [0.9026, 0.9028, 0.8904]
x_axis = np.arange(len(models))
width = 0.25

plt.figure(figsize = (12,6))
plt.bar(x_axis - width, train_scores, width, label = "Train Score")
plt.bar(x_axis, test_scores, width, label = "Test Score")
plt.bar(x_axis + width, cv_scores, width, label = "CV Mean Score")
plt.xticks(x_axis, models)
plt.ylabel("Score")
plt.title("Model Comparison")
plt.legend()
plt.ylim(0.6,1.05)
plt.show()