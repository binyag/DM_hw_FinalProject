
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
import sklearn
import subprocess
from madlan_data_prep import prepare_data
#df.groupby('type').apply(lambda x: (x['Area'] / x['room_number']).mean())
path = 'https://github.com/binyag/DM_hw_FinalProject/raw/main/output_all_students_Train_v10.xlsx'
df = pd.read_excel(path)
df = prepare_data(df)
dft_path = "https://github.com/binyag/DM_hw_FinalProject/raw/main/Dataset_for_test.xlsx"
dft = pd.read_excel(dft_path)
dft = prepare_data(dft)




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
    ('model', ElasticNetCV(cv=10,max_iter=1000, l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                           alphas=np.logspace(-5, 1, 10)))])
cv = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=13)
cv_scores = cross_val_score(pipe_preprocessing_model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
#mse KFold
kfold_mse  = np.abs(cv_scores.mean())
print(f"MSE KFold: {np.round(kfold_mse, 1)}")
# Fit the pipeline to the training data
pipe_preprocessing_model.fit(X_train, y_train)

# Predict on the test data
y_pred = pipe_preprocessing_model.predict(X_test)

# Evaluate the model performance
from sklearn.metrics import mean_absolute_error, median_absolute_error, explained_variance_score

def score_model(y_test, y_pred, model_name):
    RMSE = mean_squared_error(y_test, y_pred,squared =False )
    MSE = mean_squared_error(y_test, y_pred )
    R_squared = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    MedAE = median_absolute_error(y_test, y_pred)
    EVS = explained_variance_score(y_test, y_pred)
    print(f"Model: {model_name}")
    print(f"RMSE: {np.round(RMSE, 2)}")
    print(f"MSE: {np.round(MSE, 2)}")
    print(f"R-Squared: {np.round(R_squared, 2)}")
    print(f"Mean Absolute Error: {np.round(MAE, 2)}")
    print(f"Median Absolute Error: {np.round(MedAE, 2)}")
    print(f"Explained Variance Score: {np.round(EVS, 2)}")

score_model(y_test, y_pred, "linear_model.ElasticNetCV")


"""
הדפסת משקלים
# Get the feature names from the column transformer
feature_names = column_transformer.named_transformers_['categorical_preprocessing'].named_steps['one_hot_encoding'].get_feature_names_out(cat_cols)
feature_names = np.concatenate([num_cols, feature_names])

# Get the coefficients from the trained model
coefs = pipe_preprocessing_model.named_steps['model'].coef_

# Print the coefficients and feature names
for coef, feature in zip(coefs, feature_names):
    print(f"{feature}: {coef}")
"""
# Save the model to a file
joblib.dump(pipe_preprocessing_model, 'trained_model.pkl')
