import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# read the data
# data is from: https://www.kaggle.com/datasets/juanmerinobermejo/us-sales-cars-dataset?resource=download
data = pd.read_csv('car_sales/cars.csv')

# Drop rows with missing values
data.dropna(inplace=True)

print(data.head())

# we're not interested in the dealers
y = data['Price']

data.drop('Dealer', axis=1, inplace=True)
data.drop('Price', axis=1, inplace=True)
X = data	

print(y.head())

# change those names to ints
data['Brand'] = data['Brand'].astype('category').cat.codes
data['Model'] = data['Model'].astype('category').cat.codes

print(X.head())

# Use an 80-20 split for training and testing, use values to get rid of featurename errors
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)

# Use default settings for the RandomForestRegressor
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error (MAE): {round(mae, 2)}")

# Predict the price of a vehicle (make sure you use the right brand/model combo and status/miles combo)
brand = 28
brand_model = 531
year = 2019
status = 1
mileage = 25000
car_array = np.array([[brand, brand_model, year, status, mileage]])
prediction = model.predict(car_array)

# Print the prediction
print(f"Predicted car price for a: {brand} is: ${round(prediction[0], 2)}")