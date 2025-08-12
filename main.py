import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

def load_california_and_save(excel_path="california_housing.xlsx"):
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_excel(excel_path, index=False)
    print(f"California data saved to {excel_path}")
    return df

#  Load dataset
df = load_california_and_save()

#  Prepare features and target
X = df.drop(columns=[df.columns[-1]])
y = df[df.columns[-1]]

#  Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#  Print evaluation metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

#  Plot Actual vs Predicted scatter with ideal line
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Predictions')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red', linewidth=2, label='Ideal (y = x)')
plt.xlabel("Actual Median House Value")
plt.ylabel("Predicted Median House Value")
plt.title("Actual vs Predicted — California Housing")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
