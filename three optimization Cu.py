# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 00:39:18 2024

@author: User
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score,mean_squared_error
from skopt import BayesSearchCV
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import pandas as pd



# Define the dependent (target) and independent variables



data = pd.read_csv("data_for_submit.csv")



X = data.drop(columns=['Zn_g','Cu_g','ID_g','Lat','Lon'])  # Independent variables
y = data['Cu_g']                 # Dependent variable



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)



# Random Forest Model
def create_model(params):
    model = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        max_features=params['max_features'],
        random_state=42
    )
    return model

# 1. Grid Search Optimization
def grid_search_optimization():
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500, 600],
        'max_depth': [5, 10, 15],
        'max_features': ['sqrt', 'log2', 1.0]
    }

    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_score_

# 2. Bayesian Optimization
def bayesian_optimization():
    param_space = {
        'n_estimators': (10, 600),
        'max_depth': (2, 20),
        'max_features': ['sqrt', 'log2', 1.0]
    }

    opt = BayesSearchCV(
        RandomForestRegressor(random_state=42),
        search_spaces=param_space,
        n_iter=25,
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    opt.fit(X_train, y_train)
    return opt.best_estimator_, opt.best_score_

# 3. TPE Optimization


def tpe_optimization():
    def objective(params):
        model = create_model(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {'loss': -r2_score(y_test, y_pred), 'status': STATUS_OK}

    param_space = {
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
        'max_depth': hp.quniform('max_depth', 2, 30, 1),
        'max_features': hp.uniform('max_features', 0.1, 1.0)  # Now using a range between 0.1 and 1.0
    }

    trials = Trials()

    best = fmin(
        fn=objective,
        space=param_space,
        algo=tpe.suggest,
        max_evals=30,
        trials=trials
        
    )
    
    best_params = {
        'n_estimators': int(best['n_estimators']),
        'max_depth': int(best['max_depth']),
        'max_features': best['max_features']
    }
    
    model = create_model(best_params)
    return model, -min(trials.losses())



# Comparing the results
models = {}

# Grid Search Optimization
grid_model, grid_r2 = grid_search_optimization()
models['Grid Search'] = (grid_model, grid_r2)

# Bayesian Optimization
bayes_model, bayes_r2 = bayesian_optimization()
models['Bayesian Optimization'] = (bayes_model, bayes_r2)

# TPE Optimization
tpe_model, tpe_r2 = tpe_optimization()
models['TPE Optimization'] = (tpe_model, tpe_r2)

# Evaluate and select the best
best_method = None
best_model = None
best_r2_train = -np.inf
best_r2_test = -np.inf

for method, (model, r2_test) in models.items():
    model.fit(X_train, y_train)
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    
    print(f"{method} - Training R²: {r2_train:.4f}, Testing R²: {r2_test:.4f}")
    
    if r2_test > best_r2_test:
        best_method = method
        best_model = model
        best_r2_train = r2_train
        best_r2_test = r2_test
        best_params = model.get_params()

rmse_train = np.sqrt(mean_squared_error(y_train, best_model.predict(X_train)))
rmse_test = np.sqrt(mean_squared_error(y_test, best_model.predict(X_test)))

print(f"\nBest Method: {best_method}")
print(f"Best Training R²: {best_r2_train:.4f}")
print(f"Best Testing R²: {best_r2_test:.4f}")
print(f"Training RMSE: {rmse_train:.4f}")
print(f"Testing RMSE: {rmse_test:.4f}")
print(f"Best Parameters: {best_params}")

# Cross-validation for the best model
cv_r2_scores = cross_val_score(best_model,X_train,y_train,cv=10,scoring='r2')
cv_rmse_scores = cross_val_score(best_model,X_train,y_train,cv=10,scoring='neg_root_mean_squared_error')
cv_rmse_scores = -cv_rmse_scores
# Optionally, return the results as a dictionary or DataFrame
results = {'Fold': np.arange(1, 11),'R²': cv_r2_scores,'RMSE': cv_rmse_scores}
results_df = pd.DataFrame(results)
# Show the results in a table
print(results_df)


# Get feature importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print feature rankings
print("Feature ranking:")
for i in range(X_train.shape[1]):
    print(f"{i + 1}. Feature {X_train.columns[indices[i]]} ({importances[indices[i]]})")

# Extract the top 10 features
top_n = 20
top_features = X_train.columns[indices][:top_n]
top_importances = importances[indices][:top_n]

top_features_df = pd.DataFrame({
    'Feature': top_features,
    'Importance': top_importances
})

# Save the DataFrame to a CSV file
top_features_df.to_csv('imp_features_Cu.csv', index=False)


# write predicted data to csv

train_data = pd.DataFrame(X_train, columns=X_train.columns)
test_data = pd.DataFrame(X_test, columns=X_test.columns)

# Add the actual and predicted target values to the train and test data
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_data['Actual'] = y_train
train_data['Predicted'] = y_train_pred
train_data['DataSet'] = 'Train'  # Add identifier for the train set

test_data['Actual'] = y_test
test_data['Predicted'] = y_test_pred
test_data['DataSet'] = 'Test'  # Add identifier for the test set

# Combine the train and test data into a single DataFrame
combined_data = pd.concat([train_data, test_data], ignore_index=True)

# Write the combined data to a CSV file
combined_data.to_csv('train_test_predictions_Cu.csv', index=False)




# Best Method: TPE Optimization
# Best Training R²: 0.8874
# Best Testing R²: 0.5023
# Training RMSE: 0.4485
# Testing RMSE: 0.6847
# Best Parameters: {'bootstrap': True, 'ccp_alpha': 0.0, 'criterion': 'squared_error', 'max_depth': 13, 'max_features': 0.8374792526951379, 'max_leaf_nodes': None, 'max_samples': None, 'min_impurity_decrease': 0.0, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 852, 'n_jobs': None, 'oob_score': False, 'random_state': 42, 'verbose': 0, 'warm_start': False}
#    Fold        R²      RMSE
# 0     1  0.155596  1.163105
# 1     2  0.143722  1.116697
# 2     3  0.258291  1.252414
# 3     4  0.317614  1.046622
# 4     5  0.237557  1.277181
# 5     6  0.230682  1.428642
# 6     7  0.231745  1.223327
# 7     8  0.234576  0.981560
# 8     9  0.289094  1.009969
# 9    10  0.419429  0.965267
# Feature ranking:
# 1. Feature CI (0.15027325620551585)
# 2. Feature Altitude (0.11740644682905253)
# 3. Feature Cu_sa (0.07611667735386068)
# 4. Feature SOM (0.07566302282530406)
# 5. Feature pH (0.05061492261573547)
# 6. Feature Cd_sa (0.05046304501998293)
# 7. Feature Slope (0.04904116743219069)
# 8. Feature TWI (0.04781664986010646)
# 9. Feature Zn_sa (0.04187082052248242)
# 10. Feature Zn_s (0.041552637694263206)
# 11. Feature Temp (0.040800444178862305)
# 12. Feature Aspect (0.038784834758417)
# 13. Feature Cd_s (0.03857338479685165)
# 14. Feature Ni_s (0.037053970546946625)
# 15. Feature PSRI (0.03330860107318429)
# 16. Feature Ni_sa (0.032635044935525014)
# 17. Feature CSVI (0.02845031422434091)
# 18. Feature NDVI (0.025641622040534745)
# 19. Feature Cu_s (0.02393313708684342)


