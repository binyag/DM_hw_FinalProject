
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

def convert_value(value):
    if value == True or 'יש' in str(value) or value == 'yes' or value == 'כן' or value==1:
      return 1
    elif 'לא נגיש' not in str(value) and 'נגיש' in str(value):
      return 1
    else:
      return 0
def clean_and_convert_to_num(df, column):
  # Convert the 'Area' column to string type
  df[column] = df[column].astype(str)
  # Clean the column
  df[column] = df[column].str.replace('[^\d.]', '', regex=True)  # Remove non-digit characters except '.'
  # Convert the column to numeric
  df[column] = pd.to_numeric(df[column], errors='coerce')
  # Replace empty strings with NaN values
  df[column] = df[column].replace('', np.nan)
  # Convert the column to integer
  df[column] = df[column].astype('float64')
  
def clean_and_convert_columns(df, columns):
     for column in columns:
         df[column] = df[column].replace(['NaN', 'None', 'nan', 'none','"'] ,'')
         df[column] = df[column].replace('', np.nan)
         df[column] = df[column].str.strip()
  # Function to apply the conditions to the date values
def apply_date_conditions(date):
    # Convert the date object to a string
    date_str = date.strftime('%Y-%m-%d %H:%M:%S')

    # Convert the string back to a datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    today = datetime.now()
    # Calculate the time difference in months between the dates
    months_diff = (date_obj - today) // timedelta(days=30)

    if months_diff < 6:
        return 'less_than_6_months'
    elif 6 <= months_diff < 12:
        return 'months_6_12'
    else:
        return 'above_year'

def extract_floors(x):
  if pd.isnull(x):
    return np.nan, np.nan
  elif x == 'קומת מרתף':
    return 1, np.nan
  elif x == 'קומת קרקע':
    return 1, np.nan
  elif 'קרקע' in x:
    x = x.replace('קרקע', '1')
  parts = x.split(' ')
  if len(parts) == 2:
    floor = int(parts[1])
    total_floors = np.nan
  else:
    try:
      floor = int(parts[1])
      total_floors = int(parts[3])
    except:
      return np.nan, np.nan
  return floor, total_floors
  
def clean_and_convert_column(df, column_name):
      # Clean the specified column
      df[column_name] = df[column_name].replace(['NaN', 'None', 'nan', 'none'], '')

      # Convert the specified column to numeric
      df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('Int64')

      # Convert the specified column to integer
      df[column_name] = df[column_name].astype('Int64')
  # Define a function to remove non-Hebrew characters from a string
def remove_unnecessary_punctuation(s):
    return re.sub(r"[^א-ת\s0-9A-Za-z]", "", str(s))


# Load the data into a DataFrame
path = 'https://github.com/binyag/DM_hw_FinalProject/raw/main/output_all_students_Train_v10.xlsx'
df = pd.read_excel(path)
def prepare_data(df):
  df.columns = df.columns.str.strip()
  """תיקון עמודת עיר והסרת רווחים מיותרים"""

  df['City'] = df['City'].str.replace('נהרייה', 'נהריה')
  df['City'] = df['City'].str.replace(' שוהם', 'שוהם')
  for col in ['City', 'type','Street' ,'condition','floor_out_of']:
      if df[col].dtype == 'object':
          df[col] = df[col].str.strip()

  """תיקון עמודת מחיר למספר תקין והסרת דברים מיותרים"""
  # Convert the 'price' column to string type
  df['price'] = df['price'].astype(str)
  # Clean the 'price' column
  df['price'] = df['price'].str.replace(',', '')  # Remove commas
  df['price'] = df['price'].str.replace('[^\d.]', '', regex=True)  # Remove non-digit characters except '.'

  # Convert the 'price' column to int type
  df['price'] = pd.to_numeric(df['price'], errors='coerce').astype('Int64')
  mask = df['price'].notna()
  df = df[mask]

  """תיקון עמודת שטח הדירה"""


  clean_and_convert_to_num(df , 'Area')
  clean_and_convert_to_num(df , 'room_number')
  # Note: If you want to keep missing values (NaN) instead of converting them to 'Int64', remove the `.astype('Int64')` part

  """תיקון לעמודת מספר תמונות ומספר רחוב"""
  # Assuming you have a DataFrame called 'df' with the 'num_of_images' and 'number_in_street' columns
  clean_and_convert_column(df, 'num_of_images')
  clean_and_convert_column(df, 'number_in_street')

  """## תיקון בעמודת של קומות"""


      # Apply the function to the 'floor_out_of' column to create the new columns
  df[['floor', 'total_floors']] = df['floor_out_of'].apply(extract_floors).apply(pd.Series)
  mask = df['total_floors'] < df['floor']
  df.loc[mask, ['floor', 'total_floors']] = df.loc[mask, ['total_floors', 'floor']].values

  """
  המרת עמודה
  "entranceDate "
  """
  # Define the conditions and corresponding replacements
  conditions = {
      'גמיש': 'flexible',
      'גמיש ': 'flexible',
      'לא צויין': 'not_defined',
      'מיידי': 'less_than_6_months'
  }

  # Apply the conditions to the 'entranceDate' column
  df['entrance_date'] = df['entranceDate'].replace(conditions)

  # Identify date values and apply the date conditions
  date_mask = pd.to_datetime(df['entrance_date'], errors='coerce').notna()
  df.loc[date_mask, 'entrance_date'] = df.loc[date_mask, 'entrance_date'].apply(apply_date_conditions)

  """הסרת ביטויים לערכים ריקים מהעמודה אזור בעיר ורחוב"""
  # Apply the function to the 'Street','description','city_area' columns of the DataFrame
  for column in['Street','description','city_area']:
    df[column] = df[column].apply(remove_unnecessary_punctuation)

 
  columns_to_clean = ['Street','description', 'city_area']
  clean_and_convert_columns(df, columns_to_clean)

  """תיקון עמודת מצב הדירה"""

  # Replace values in the 'condition' column
  df['condition'] = df['condition'].replace({'דורש שיפוץ': 'ישן', 'None': ''})
### תיקון לכל העמודות עם ה"כן/לא יש אין" וכו

  columns_to_convert = ['hasElevator', 'hasParking', 'hasBars', 'hasStorage',
                        'hasAirCondition', 'hasBalcony', 'hasMamad', 'handicapFriendly']
  for column in columns_to_convert:
    df[column] = df[column].apply(convert_value)

  conditions = {'נחלה': 'מגרש',
                    "דו משפחתי" : "קוטג'",
                    'בניין' : 'דירה',
                    "קוטג' טורי": "קוטג'",
                    'מיני פנטהאוז' : 'דירת גג',
                    'דירת נופש' : 'דירה',
                    "טריפלקס": "בית פרטי"}
  df['type'] = df['type'].replace(conditions)

  df['furniture'] = df['furniture'].replace('אין', 'לא צויין')


  df = df.drop_duplicates(subset=['City', 'type', 'room_number', 'Area', 'Street', 'number_in_street', 'price'])
  df.loc[df['Area'] > 500, 'Area'] = np.nan
  df['area_room_ratio'] = df['Area'] / df['room_number']
  df = df[(df['area_room_ratio'] >= 18)]
  df['big_ratio'] = np.where(df['area_room_ratio'] >= 40, 1, 0)

  # Delete rows where Price is greater than 15 million
  df = df[df['price'] <= 15000000]
  df = df[df['room_number'] < 10]
  # Remove rows where the city is 'Givat Shmuel' and the price is less than a million
  
  df = df.loc[(df['City'] != 'גבעת שמואל') | (df['price'] >= 1000000)]
  df = df.loc[(df['City'] != 'זכרון יעקב') | (df['price'] >= 1000000)]
  df = df.loc[(df['City'] != 'כפר סבא') | (df['price'] >= 1000000)]
  df = df.loc[(df['City'] != 'הרצליה') | (df['price'] >= 1000000)]
    
  df = df.loc[(df['City'] != 'דימונה') | (df['price'] < 6000000)]
  df = df.loc[(df['City'] != 'ראשון לציון') | (df['price'] < 5000000)]
  return df

df  = prepare_data(df)
