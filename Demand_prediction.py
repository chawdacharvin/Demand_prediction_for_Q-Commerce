import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_excel("C:/Users/HP/Desktop/RM&P/cleaned_data.xlsx")

print(data.head())
print(data.info())

data['discounted_price'] = data['mrp'] * (1 - data['discountPercent'] / 100)  # Create discounted price column
data['total_weight'] = data['availableQuantity'] * data['weightInGms']        # Total weight for stock
data['is_stock_out'] = data['outOfStock'].apply(lambda x: 1 if x == 'Yes' else 0)  # Binary stock-out feature


data.fillna(0, inplace=True)

features = ['mrp', 'discountPercent', 'availableQuantity', 'discounted_price', 'total_weight', 'is_stock_out']
target = 'quantity'


X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)


xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)


rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_predictions))

rf_r2 = r2_score(y_test, rf_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)


print("Random Forest RMSE:", rf_rmse)
print("Random Forest R^2:", rf_r2)

print("XGBoost RMSE:", xgb_rmse)
print("XGBoost R^2:", xgb_r2)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=y_test, y=rf_predictions, alpha=0.6)
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=xgb_predictions, alpha=0.6, color='orange')
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")

plt.tight_layout()
plt.show()
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist,
                                   n_iter=50, cv=3, scoring='neg_mean_squared_error',
                                   random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

best_rf_model = random_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test)
best_rf_r2 = r2_score(y_test, best_rf_predictions)
print("Best Random Forest R^2 after tuning:", best_rf_r2)