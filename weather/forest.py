import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# read the data
# data is from: https://data.virginia.gov/dataset/weather-daily-summaries/resource/7fb4a20d-5606-4230-9633-e05fd55918e5
data = pd.read_csv('source.csv')

# Drop rows with missing values
data = data.dropna()

# This example will only be looking at avg_max temp on a day, you can change the features for min/avg/wind/precip/snow as you please
X = data.drop(columns=['temp_max', 'wind_spd_avg', 'temp_max', 'temp_min', 'temp_avg', 'precip', 'snow'])	
y = data['temp_max']

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

# Predict the target value on a day, change day here to change the prediction below
day = 87
day_array = np.array([[day]])
prediction = model.predict(day_array)

# Print the prediction
print(f"Predicted temperature high on day: {day} is: {round(prediction[0], 2)}° F")