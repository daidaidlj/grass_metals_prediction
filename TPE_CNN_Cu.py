#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 11:44:09 2024

@author: daidai
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from hyperopt import Trials, STATUS_OK, tpe
from hyperopt import fmin, hp


# Step 1: defien the training and testing data



data = pd.read_csv("data_for_submit.csv")


# Define the dependent (target) and independent variables
X = data.drop(columns=['Zn_g','Cu_g','ID_g','Lat','Lon'])  # Independent variables
y = data['Cu_g']                 # Dependent variable

# Split the data into training and test sets


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15,random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data to fit into CNN (samples, time steps, features)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Define the objective function for hyperparameter optimization
def objective(params):
    model = Sequential([
        Conv1D(filters=int(params['filters1']), kernel_size=int(params['kernel_size1']), activation='relu', input_shape=(X_train.shape[1], 1)),
        Dropout(params['dropout1']),
        Conv1D(filters=int(params['filters2']), kernel_size=int(params['kernel_size2']), activation='relu'),
        Dropout(params['dropout2']),
        Flatten(),
        Dense(int(params['dense_units']), activation='relu'),
        Dropout(params['dropout3']),
        Dense(1)  # Output layer for regression
    ])

    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse')

    model.fit(X_train, y_train, epochs=int(params['epochs']), verbose=0, validation_split=0.1)
    


    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
    return {'loss': rmse, 'status': STATUS_OK, 'model': model}

# Define the hyperparameter search space
space = {
    'filters1': hp.quniform('filters1', 16, 128, 16),
    'kernel_size1': hp.quniform('kernel_size1', 2, 5, 1),
    'filters2': hp.quniform('filters2', 16, 128, 16),
    'kernel_size2': hp.quniform('kernel_size2', 2, 5, 1),
    'dense_units': hp.quniform('dense_units', 32, 256, 32),
    'dropout1': hp.uniform('dropout1', 0.1, 0.5),
    'dropout2': hp.uniform('dropout2', 0.1, 0.5),
    'dropout3': hp.uniform('dropout3', 0.1, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'epochs': hp.quniform('epochs', 50, 200, 50)
}

# Run the optimization
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=30, trials=trials)

# Print the best hyperparameters
print("Best Hyperparameters:", best)

# Train the model with the best hyperparameters
best_model = trials.best_trial['result']['model']

# Evaluate the model on the test set
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f"R² (Training): {r2_train}")
print(f"RMSE (Training): {rmse_train}")
print(f"R² (Test): {r2_test}")
print(f"RMSE (Test): {rmse_test}")


#Best Hyperparameters: {'dense_units': 224.0, 'dropout1': 0.1459045885298772, 'dropout2': 0.4202075239727384, 'dropout3': 0.10262610971799664, 'epochs': 100.0, 'filters1': 80.0, 'filters2': 96.0, 'kernel_size1': 2.0, 'kernel_size2': 3.0, 'learning_rate': 0.00021495177154347728}
# R² (Training): 0.5210628031737214
# RMSE (Training): 0.9247936459793185
# R² (Test): 0.4047657856822634
# RMSE (Test): 0.7487816013169766
