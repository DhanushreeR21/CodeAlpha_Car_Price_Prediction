
import pandas as pd
import os

# Dataset path
dataset_path = r"C:\Users\dhanu\.cache\kagglehub\datasets\nehalbirla\vehicle-dataset-from-cardekho\versions\4\car data.csv"

# Load dataset
df = pd.read_csv(dataset_path)

# Display first five rows
print("First Five Rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Create car age feature
df["Car_Age"] = 2026 - df["Year"]

# Remove unnecessary columns
df.drop(["Car_Name", "Year"], axis=1, inplace=True)

# Convert categorical columns into numerical values
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True,
    dtype=int
)

print("\nCleaned Dataset:")
print(df.head())

print("\nFinal Dataset Shape:")
print(df.shape)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Separate input features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
rf_model.fit(X_train, y_train)

print("\nBoth models trained successfully!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Make predictions
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Function to evaluate model
def evaluate_model(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    print(f"\n{name} Results:")
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 2))

# Evaluate both models
evaluate_model("Linear Regression", y_test, lr_predictions)
evaluate_model("Random Forest", y_test, rf_predictions)

import matplotlib.pyplot as plt

# 1. Actual vs Predicted Car Prices
plt.figure(figsize=(8, 5))

plt.scatter(y_test, lr_predictions, alpha=0.7)

plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Prices (Linear Regression)")

plt.savefig("actual_vs_predicted.png")
plt.show()

# 2. Model Performance Comparison
models = ["Linear Regression", "Random Forest"]
r2_scores = [
    r2_score(y_test, lr_predictions),
    r2_score(y_test, rf_predictions)
]

plt.figure(figsize=(7, 5))

plt.bar(models, r2_scores)

plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)

plt.savefig("model_comparison.png")
plt.show()

print("\nGraphs saved successfully!")

# 3. Interactive Car Price Prediction

print("\n===== CAR PRICE PREDICTION =====")

present_price = float(input("Enter current showroom price (in lakhs): "))
kms_driven = int(input("Enter kilometers driven: "))
car_age = int(input("Enter car age (in years): "))
owner = int(input("Enter number of previous owners (0, 1, 2, or 3): "))

fuel = input("Enter fuel type (Petrol/Diesel/CNG): ").strip().capitalize()
seller = input("Enter seller type (Dealer/Individual): ").strip().capitalize()
transmission = input("Enter transmission (Manual/Automatic): ").strip().capitalize()

# Validate categorical inputs
if fuel not in ["Petrol", "Diesel", "Cng"]:
    raise ValueError("Invalid fuel type!")

if seller not in ["Dealer", "Individual"]:
    raise ValueError("Invalid seller type!")

if transmission not in ["Manual", "Automatic"]:
    raise ValueError("Invalid transmission!")

# Prepare input features
new_car = pd.DataFrame([{
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Owner": owner,
    "Car_Age": car_age,
    "Fuel_Type_Diesel": int(fuel == "Diesel"),
    "Fuel_Type_Petrol": int(fuel == "Petrol"),
    "Seller_Type_Individual": int(seller == "Individual"),
    "Transmission_Manual": int(transmission == "Manual")
}])

# Match training feature order
new_car = new_car.reindex(columns=X.columns, fill_value=0)

# Predict price using Linear Regression
predicted_price = lr_model.predict(new_car)

print("\nEstimated Selling Price: ₹", round(predicted_price[0], 2), "Lakhs")

