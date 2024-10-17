#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 22:58:50 2024

@author: daidai
"""

import optuna
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd




# Define the dependent (target) and independent variables


data = pd.read_csv("data_for_submit.csv")


# Define the dependent (target) and independent variables
X = data.drop(columns=['Zn_g','Cu_g','ID_g','Lon','Lat'])  # Independent variables
y = data['Zn_g']                 # Dependent variable

# Split the data into training and test sets



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=57)


# Step 2: Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    param = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
    }
    
    # Create the model with suggested hyperparameters
    model = xgb.XGBRegressor(**param, n_jobs=-1, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predict and calculate RMSE
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    
    return rmse

# Step 3: Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50)

# Print the best parameters
print("Best parameters found: ", study.best_params)
print("Best RMSE: ", study.best_value)

# Step 4: Train the best model on the full training data
best_model = xgb.XGBRegressor(**study.best_params, n_jobs=-1, random_state=57)
best_model.fit(X_train, y_train)

rmse_train = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
rmse_test = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
r2_train = r2_score(y_train, best_model.predict(X_train))
r2_test = r2_score(y_test, best_model.predict(X_test))


print(f"Best Training R²: {r2_train:.4f}")
print(f"Best Testing R²: {r2_test:.4f}")
print(f"Training RMSE: {rmse_train:.4f}")
print(f"Testing RMSE: {rmse_test:.4f}")



# Best parameters found:  {'n_estimators': 168, 'learning_rate': 0.03421468897872913, 'max_depth': 6, 'min_child_weight': 1, 'subsample': 0.8567990728541679, 'colsample_bytree': 0.9226490356149654, 'gamma': 0.9983099135326874, 'reg_alpha': 0.0003089240232465428, 'reg_lambda': 0.0003875091313927013}
# Best RMSE:  6.507317409102073
# Best Training R²: 0.9671
# Best Testing R²: 0.4916
# Training RMSE: 1.8415
# Testing RMSE: 6.6739
