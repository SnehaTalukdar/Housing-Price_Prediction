import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Loading the raw housing dataset
data = pd.read_csv("Housing.csv")

# Normalizing column names 
data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]

# Quick replacement for categorical text fields with numbers

category_map = {
    'yes': 1,
    'no': 0,
    'unfurnished': 0,
    'semi-furnished': 1,
    'furnished': 2
}
data.replace(category_map, inplace=True)

# Export cleaned dataset so we don’t lose this version
data.to_csv("cleaned_housing_data.csv", index=False)
print(" Cleaned dataset saved as 'cleaned_housing_data.csv'\n")
print(data.head()) 

# Simple Linear Regression: Area V/S Price 

#  price scales with area
x_area = data[['area']]  
y_price = data['price']

x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(
    x_area, y_price, test_size=0.2, random_state=42
)

area_model = LinearRegression()
area_model.fit(x_train_a, y_train_a)

predicted_prices_area = area_model.predict(x_test_a)

# Printing out results
print("\n Simple Linear Regression: Area vs Price")
print(f"Intercept: {area_model.intercept_:.2f}")
print(f"Coefficient: {area_model.coef_[0]:.2f}")
print(f"MAE: {mean_absolute_error(y_test_a, predicted_prices_area):,.2f}")
print(f"MSE: {mean_squared_error(y_test_a, predicted_prices_area):,.2f}")
print(f"R² Score: {r2_score(y_test_a, predicted_prices_area):.2f}")

# Plotting the predicted line after it fits the actual prices
plt.figure(figsize=(8, 6))
plt.scatter(x_test_a, y_test_a, color='skyblue', label='Actual Prices')
plt.plot(x_test_a, predicted_prices_area, color='red', label='Prediction Line')
plt.title("Simple Linear Regression: Area vs Price")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

#  Multiple Linear Regression 

X = data.drop(columns='price')  # All features except price
y = data['price']

x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(
    X, y, test_size=0.2, random_state=42
)

multi_model = LinearRegression()
multi_model.fit(x_train_m, y_train_m)
predictions_m = multi_model.predict(x_test_m)

# Results 
print("\n Multiple Linear Regression Results")
print(f"Intercept: {multi_model.intercept_:.2f}")
print("Feature Coefficients:")
for feat, val in zip(X.columns, multi_model.coef_):
    print(f"{feat}: {val:.2f}")

print(f"\nMAE: {mean_absolute_error(y_test_m, predictions_m):,.2f}")
print(f"MSE: {mean_squared_error(y_test_m, predictions_m):,.2f}")
print(f"R² Score: {r2_score(y_test_m, predictions_m):.2f}")

# Residuals for any missing patterns of model
resids = y_test_m - predictions_m

plt.figure(figsize=(8, 6))
sns.scatterplot(x=predictions_m, y=resids, color='purple')
plt.axhline(0, linestyle='--', color='black')
plt.title("Residuals Plot (Multiple Linear Regression)")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()

#  correlation heatmap to see how things relate to each other
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
