#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 22:58:50 2024

@author: daidai
"""

import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd


# Step 1: defien the training and testing data

file_path = "E:/grass_three_places/submit to EST_windows/first_revision/data_10162024/"

data = pd.read_csv(file_path +"data_for_submit.csv")


# Define the dependent (target) and independent variables
X = data.drop(columns=['Zn_g','Cu_g','ID_g','Lat','Lon'])  # Independent variables
y = data['Cu_g']                 # Dependent variable

# Split the data into training and test sets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,random_state=42)




# Step 2: Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    param = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'gamma': trial.suggest_uniform('gamma', 0, 1.0),
#        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1),
#       'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1)
    }
    
    # Create the model with suggested hyperparameters
    model = xgb.XGBRegressor(**param, n_jobs=-1)
    
    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Predict and calculate RMSE
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    
    return rmse

# Step 3: Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=30)

# Print the best parameters
print("Best parameters found: ", study.best_params)
# Step 4: Train the best model on the full training data
best_model = xgb.XGBRegressor(**study.best_params, n_jobs=-1, random_state=42)
best_model.fit(X_train, y_train)


rmse_train = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
rmse_test = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))
r2_train = r2_score(y_train, best_model.predict(X_train))
r2_test = r2_score(y_test, best_model.predict(X_test))


print(f"Best Training R²: {r2_train:.4f}")
print(f"Best Testing R²: {r2_test:.4f}")
print(f"Training RMSE: {rmse_train:.4f}")
print(f"Testing RMSE: {rmse_test:.4f}")


# Best parameters found:  {'n_estimators': 227, 'learning_rate': 0.06275423665932531, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.6910077132031013, 'colsample_bytree': 0.6964921091764134, 'gamma': 0.7053778620567941}
# Best Training R²: 0.8275
# Best Testing R²: 0.4895
# Training RMSE: 0.5550
# Testing RMSE: 0.6934