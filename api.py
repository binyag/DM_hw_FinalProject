import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form
    
    City = str(data['City'])
    type1 = str(data['type'])
    condition = str(data['condition'])
    city_area = str(data['city_area'])
    room_number = float(data['room_number'])
    Area = int(data['Area'])
    hasElevator = int(data.get('hasElevator', 0))
    hasParking = int(data.get('hasParking', 0))
    hasBars = int(data.get('hasBars', 0))
    hasStorage = int(data.get('hasStorage', 0))
    hasAirCondition = int(data.get('hasAirCondition', 0))
    hasBalcony = int(data.get('hasBalcony', 0))
    hasMamad = int(data.get('hasMamad', 0))
    handicapFriendly = int(data.get('handicapFriendly', 0))
    floor = int(data['floor'])
    big_ratio = np.where((Area / room_number) >= 40, 1, 0)
    # Create a feature array
    
    data = {'City': City, 'type': type1, 'condition': condition,
            'city_area': city_area, 'room_number': room_number, 'Area': Area,
            'hasElevator': hasElevator, 'hasParking': hasParking, 'hasBars':hasBars,
            'hasStorage': hasStorage, 'hasAirCondition': hasAirCondition,
            'hasBalcony': hasBalcony, 'hasMamad': hasMamad, 'handicapFriendly': handicapFriendly
            , 'floor': floor, 'big_ratio':big_ratio }
    df = pd.DataFrame(data, index=[0])
    # Make a prediction
    y_pred =round( model.predict(df)[0])

    # Return the predicted price
    return render_template('index.html', price=y_pred)

if __name__ == '__main__':
    app.run()