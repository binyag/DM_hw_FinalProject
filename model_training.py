
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
import seaborn as sns
from datetime import datetime, timedelta
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import subprocess
from madlan_data_prep import prepare_data
#df.groupby('type').apply(lambda x: (x['Area'] / x['room_number']).mean())
path = 'https://github.com/binyag/DM_hw_FinalProject/raw/main/output_all_students_Train_v10.xlsx'
df = pd.read_excel(path)
df = prepare_data(df)
dft_path = "https://github.com/binyag/DM_hw_FinalProject/raw/main/Dataset_for_test.xlsx"
dft = pd.read_excel(dft_path)
dft = prepare_data(dft)
"""## המודל


מילוי ערכים חסרים בעמודת מסםר חדרים באמצעות נוסחה של אזור
"""

# Define the target column
target_col = 'price'
column_to_model  = ['City', 'type', 'room_number', 'Area', 'city_area', 'hasElevator',
       'hasParking', 'hasBars', 'hasStorage', 'condition', 'hasAirCondition',
       'hasBalcony', 'hasMamad', 'handicapFriendly', 'floor',  'big_ratio']
# Split the data into features and targets
X_train = df[column_to_model]
y_train = df[target_col]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_test = dft[column_to_model]
y_test = dft[target_col]
# Categorical columns
cat_cols = ['City', 'type' , 'hasElevator',
            'hasParking',  'hasStorage', 'condition', 
       'hasBalcony', 'hasMamad', 'handicapFriendly','city_area', 'big_ratio']
columns_to_fill = ['city_area','condition']

# Fill missing values in selected categorical columns
X_train[columns_to_fill] = X_train[columns_to_fill].fillna('לא צויין')
X_test[columns_to_fill] = X_test[columns_to_fill].fillna('לא צויין')


cat_pipeline = Pipeline([
    ('one_hot_encoding', OneHotEncoder(sparse=False, handle_unknown='ignore',drop='if_binary'))
])

# Numerical columns
num_cols = ['room_number', 'Area', 'floor']
"""
def custom_fill_transformer(X):
    X_filled = X.copy()
    X_filled['room_number'] = X_filled['room_number'].fillna(X_filled['Area'].apply(lambda x: np.round(x / 26 * 2) / 2))
    X_filled['Area'] = X_filled['Area'].fillna(X_filled['room_number'].apply(lambda x: np.round(x * 26)))
    return X_filled
num_pipeline = Pipeline([
    ('custom_fill', FunctionTransformer(custom_fill_transformer)),
    ('numerical_imputation', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler())
])
"""
X_train['room_number'] = X_train['room_number'].fillna(X_train['Area'].apply(lambda x: np.round(x / 26 * 2) / 2))
X_train['Area'] = X_train['Area'].fillna(X_train['room_number'].apply(lambda x: np.round(x * 26)))
num_pipeline = Pipeline([('numerical_imputation', SimpleImputer(strategy='median')),
    ('scaling', StandardScaler())
])

# ColumnTransformer
column_transformer = ColumnTransformer([
    ('numerical_preprocessing', num_pipeline, num_cols),
    ('categorical_preprocessing', cat_pipeline, cat_cols)
], remainder='drop')

# Define the pipeline
pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNetCV(cv=5,max_iter=1000, l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                           alphas=np.logspace(-5, 1, 10)))])

# Fit the pipeline to the training data
pipe_preprocessing_model.fit(X_train, y_train)

# Predict on the test data
y_pred = pipe_preprocessing_model.predict(X_test)

# Evaluate the model performance
def score_model(y_test, y_pred, model_name):
    RMSE = mean_squared_error(y_test, y_pred,squared =False )
    #RMSE = np.sqrt(MSE)
    R_squared = r2_score(y_test, y_pred)
    print(f"Model: {model_name}, RMSE: {np.round(RMSE, 2)}, R-Squared: {np.round(R_squared, 2)}")

score_model(y_test, y_pred, "linear_model.ElasticNetCV")

# Evaluate the model performance
def score_model(y_test, y_pred, model_name):
    RMSE = mean_squared_error(y_test, y_pred,squared =False )
    #RMSE = np.sqrt(MSE)
    R_squared = r2_score(y_test, y_pred)
    print(f"Model: {model_name}, RMSE: {np.round(RMSE, 2)}, R-Squared: {np.round(R_squared, 2)}")


# Get the feature names from the column transformer
feature_names = column_transformer.named_transformers_['categorical_preprocessing'].named_steps['one_hot_encoding'].get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, feature_names])

# Get the coefficients from the trained model
coefs = pipe_preprocessing_model.named_steps['model'].coef_

# Print the coefficients and feature names
for coef, feature in zip(coefs, feature_names):
    print(f"{feature}: {coef}")

"""קוד להצגת כלל המודלים שרצים בבקרוס וולדציה"""

# from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV

# # Define a list of models
# models = [
#     ('ElasticNetCV', ElasticNetCV(cv=5,max_iter=1000, l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
#                                   alphas=np.logspace(-5, 1, 10))),
#     ('RidgeCV', RidgeCV(cv=5)),
#     ('LassoCV', LassoCV(cv=5))
#     # Add more models if desired
# ]

# # Create a dictionary to store the model errors
# model_errors = {}

# # Fit and evaluate each model in the pipeline
# for model_name, model in models:
#     pipe_preprocessing_model.set_params(model=model)  # Set the current model in the pipeline
#     pipe_preprocessing_model.fit(X_train, y_train)  # Fit the pipeline to the training data
#     y_pred = pipe_preprocessing_model.predict(X_test)  # Predict on the test data
#     RMSE = mean_squared_error(y_test, y_pred, squared=False)  # Calculate the RMSE
#     model_errors[model_name] = RMSE  # Store the RMSE in the dictionary

# # Print the model errors
# for model_name, error in model_errors.items():
#     print(f"Model: {model_name}, RMSE: {error}")

# Save the model to a file
joblib.dump(pipe_preprocessing_model, 'model.pkl')
score_model(y_test, y_pred, "linear_model.ElasticNetCV")
