import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Create dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Scores': [20, 35, 50, 55, 60, 75, 85, 88, 95]
}

df = pd.DataFrame(data)

# Step 2: Print the dataset
print(df)

# Step 3: Plot scatter chart
plt.scatter(df['Hours'], df['Scores'], color='blue')
plt.title('Study Hours vs Score')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.grid(True)
plt.show()

# Step 4: Prepare data for training
X = df[['Hours']]  # 2D array
y = df['Scores']   # 1D array

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 6: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ Step 7: Predict the score for 6.5 study hours
predicted_score = model.predict([[6.5]])
print(f"\n📘 Predicted score for studying 6.5 hours: {predicted_score[0]:.2f}")

# ✅ Step 8: Plot regression line with original data
line = model.coef_ * df['Hours'] + model.intercept_

plt.scatter(df['Hours'], df['Scores'], color='blue', label='Actual')
plt.plot(df['Hours'], line, color='red', label='Regression Line')
plt.scatter(6.5, predicted_score, color='green', label='Predicted (6.5 hrs)', zorder=5)

plt.title('Study Hours vs Score (with Prediction)')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.show()
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 9: Predict scores for test data
y_pred = model.predict(X_test)

# Step 10: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")

import joblib

# Save the model
joblib.dump(model, 'score_predictor_model.pkl')

# Save the vectorized input shape (optional)
print("\n✅ Model saved as 'score_predictor_model.pkl'")
