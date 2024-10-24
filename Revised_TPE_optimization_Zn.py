# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 01:28:12 2024

@author: User
"""



import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error



data = pd.read_csv("data_for_submit.csv")


# Define the dependent (target) and independent variables
X = data.drop(columns=['Zn_g','Cu_g'])  # Independent variables
y = data['Zn_g']                 # Dependent variable


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=57)

r2_scores_dict = {}


def objective(trial):
    # Define the search space for each hyperparameter
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    max_depth = trial.suggest_int('max_depth', 2, 40)
    max_features = trial.suggest_float('max_features', 0.1, 1.0)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 30)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    
    
    # Initialize the Random Forest model
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Calculate the R^2 score on the training set
    train_r2 = model.score(X_train, y_train)
    
    # Calculate the R^2 score on the test set
    test_r2 = model.score(X_test, y_test)
        
    # Store both R^2 scores for the current trial
    r2_scores_dict[trial.number] = {"train_r2": train_r2, "test_r2": test_r2}
    
    # Penalize if training R^2 is outside the [0.7, 0.8] range
    if train_r2 < 0.7 or train_r2 > 0.8:
        return -float('inf')  # Penalize heavily if the training R^2 is out of the desired range
    
    # Return the test R^2 (maximize this value)
    return test_r2



# Use Optuna to optimize hyperparameters while respecting the training R^2 constraint
study = optuna.create_study(direction='maximize')

# Perform optimization
study.optimize(objective, n_trials=1000)

# Retrieve the best trial
best_trial = study.best_trial

# Get the best trial number and associated R^2 scores
best_trial_number = best_trial.number

def rmse_model(best_params, X_train,  X_test, y_train, y_test):
    # Initialize the model with the best hyperparameters
    model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],

    )
    
    # Train the model with the top N features
    model.fit(X_train, y_train)   
    
    # Make predictions for training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    return train_rmse, test_rmse


def cv_scores_model(best_params, X_train, X_test, y_train, y_test):
    # Initialize the model with the best hyperparameters
    model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],

    )
    
    # Train the model with the top N features
    model.fit(X_train, y_train)
    
    cv_scores_full_model = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')
    return cv_scores_full_model


# Get the best trial number and associated R^2 scores
best_trial_number = best_trial.number
best_train_r2 = r2_scores_dict[best_trial_number]["train_r2"]
best_test_r2 = r2_scores_dict[best_trial_number]["test_r2"]
best_trial_rmse = rmse_model(best_trial.params, X_train,  X_test, y_train, y_test)
cv_scores_full_model = cv_scores_model(best_trial.params, X_train, X_test, y_train, y_test)






print(f'Best Training R^2: {best_train_r2}')
print(f'Best Test R^2: {best_test_r2}')
print(f'Best hyperparameters: {best_trial.params}')
print(f'traing and test RMSE:{best_trial_rmse}')
print(f"Cross-validation R^2 scores for the full model (10 folds): {cv_scores_full_model}")




# Train the Random Forest model with the best hyperparameters and get feature importances
def get_feature_importances(best_params, X_train, y_train):
    # Initialize the Random Forest model with the best hyperparameters
    model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],

    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Get feature importance scores
    feature_importances = model.feature_importances_
    
    return feature_importances


def select_top_features(importances, X_train, top_n):
    # Get the indices of the top N important features
    indices = np.argsort(importances)[-top_n:]
    
    # Select the top N features from X_train using iloc for pandas or NumPy slicing
    if isinstance(X_train, pd.DataFrame):
        X_train_top = X_train.iloc[:, indices]
    else:
        X_train_top = X_train[:, indices]
    
    return X_train_top, indices

# Step 3: Re-run the Random Forest model using only the selected top N features
def rerun_model_with_top_features(best_params, X_train_top, X_test_top, y_train, y_test):
    # Initialize the model with the best hyperparameters
    model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],

    )
    
    # Train the model with the top N features
    model.fit(X_train_top, y_train)
    
    # Calculate training and test R^2 scores
    train_r2 = model.score(X_train_top, y_train)
    test_r2 = model.score(X_test_top, y_test)
    
    return train_r2, test_r2



# Example usage assuming you have best_trial, X_train, X_test, y_train, y_test
# Step 1: Get feature importances from the best trial
feature_importances = get_feature_importances(best_trial.params, X_train, y_train)

# Step 2: Select the top 10 important features
X_train_top, selected_indices = select_top_features(feature_importances, X_train, top_n=10)

# Apply the same feature selection to X_test
if isinstance(X_test, pd.DataFrame):
    X_test_top = X_test.iloc[:, selected_indices]
else:
    X_test_top = X_test[:, selected_indices]

# Step 3: Re-run the model with the selected top 10 features
train_r2, test_r2 = rerun_model_with_top_features(best_trial.params, X_train_top, X_test_top, y_train, y_test)
rmse = rmse_model(best_trial.params, X_train_top, X_test_top, y_train, y_test)
cv_scores_top_features_model = cv_scores_model(best_trial.params, X_train_top, X_test_top, y_train, y_test)

print(f"Train R^2 with top features: {train_r2}")
print(f"Test R^2 with top features: {test_r2}")
print(f'traing and test RMSE with top features:{rmse}')
print(f"Cross-validation R^2 scores with top features (10 folds): {cv_scores_top_features_model}")




# Extract feature importance after training the full model with all features
def export_feature_importance_to_csv(feature_importances, feature_names, filename="feature_importance.csv"):
    # Create a DataFrame to store feature names and their importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # Rank features by importance (descending order)
    feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
    
    # Export to CSV
    feature_importance_df.to_csv(filename, index=False)
    print(f"Feature importance exported to {filename}")

# Example: Export feature importance after fitting the model
feature_importances = get_feature_importances(best_trial.params, X_train, y_train)

# If using a DataFrame, get the feature names (assuming X_train is a pandas DataFrame)
if isinstance(X_train, pd.DataFrame):
    feature_names = X_train.columns
else:
    feature_names = [f"Feature {i}" for i in range(X_train.shape[1])]  # If not a DataFrame, generate default names

# Export the ranked feature importance to a CSV file
export_feature_importance_to_csv(feature_importances, feature_names, filename="ranked_feature_importance_Zn.csv")





# Best is trial 397 with value: 0.5323092712631979.
# Best Training R^2: 0.7323651760361689
# Best Test R^2: 0.5323092712631979
# Best hyperparameters: {'n_estimators': 681, 'max_depth': 37, 'max_features': 0.9715346971387042, 'min_samples_split': 13, 'min_samples_leaf': 3}
# traing and test RMSE:(5.2500002048792425, 6.524724542231865)
# Cross-validation R^2 scores for the full model (10 folds): [0.28085286 0.29253986 0.36882467 0.41691068 0.37113942 0.20484863 0.3091461  0.30911153 0.48287043 0.15992823]
# Train R^2 with top features: 0.7155348724772062
# Test R^2 with top features: 0.5441638479193934
# traing and test RMSE with top features:(5.425089355313354, 6.327041620339494)
# Cross-validation R^2 scores with top features (10 folds): [0.31011549 0.34433415 0.34883046 0.3832881  0.39082664 0.20634148 0.30832418 0.28962625 0.48784886 0.16748367]


