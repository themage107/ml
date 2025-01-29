import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# read the data
# data is from: https://archive.ics.uci.edu/dataset/186/wine+quality

# pick one
data = pd.read_csv('wine_quality/winequality-red.csv')
# data = pd.read_csv('wine_quality/winequality-white.csv')

# Drop rows with missing values
data.dropna(inplace=True)


# Lets use all the data to find quality
X = data.drop(columns=['quality'])	
y = data['quality']

# Use an 80-20 split for training and testing, use values to get rid of featurename errors
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=0)

# Use default settings for the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=0)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMean Absolute Error (MAE): {round(mae, 2)}")

# Predict the price of a vehicle (make sure you use the right brand/model combo and status/miles combo)
fixed_acidity = 6.7
volatile_acidity = 0.26
citric_acid = 0.06
residual_sugar = 1.6
chlorides = 0.069
free_sulfur_dioxide = 15
total_sulfur_dioxide = 59
density = 0.994
pH = 3.7
sulphates = 0.46
alcohol = 10

wine_array = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
prediction = model.predict(wine_array)

# Print the prediction
print(f"Predicted quality for wine is: {round(prediction[0], 2)}")