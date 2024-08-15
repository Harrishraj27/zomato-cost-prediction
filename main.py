import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("zomato.csv")
data.head()

def ratings(x):
    if (x == "NEW" or x == "-"):
        return np.nan
    else:
        x = str(x).split("/")
        x = x[0]
        return float(x)

data["rate"] = data["rate"].apply(ratings)

# Drop duplicates
data.drop_duplicates(inplace=True)

# Impute missing values
data['rate'].fillna(data['rate'].mean(), inplace=True)
data.rename(columns={"approx_cost(for two people)": "Cost_for_two", "listed_in(type)": "Type", "rate": "Ratings"}, inplace=True)
data["Cost_for_two"] = data["Cost_for_two"].str.replace(",", "").astype(float)
data["votes"] = data["votes"].astype(float)
data["Cost_for_two"].fillna(data['Cost_for_two'].mean(), inplace=True)

# Handle categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['online_order', 'book_table', 'location', 'rest_type', 'dish_liked', 'cuisines', 'Type']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col].astype(str))

data.drop(['name'], axis=1,inplace=True)
X = data.drop('Cost_for_two', axis=1)
y = data['Cost_for_two']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

## Evaluation Metrics

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

# Calculate R-squared (R2) score
r2 = r2_score(y_test, y_pred)
print("R-squared (R2) Score:", r2)
