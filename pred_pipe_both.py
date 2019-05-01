# Importing libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import pickle
import xgboost as xgb

# Converting to onehot
def onehot(df):
    one_hot = pd.get_dummies(df['etype'])
    df = df.drop('etype',axis = 1)
    df = df.join(one_hot)

    one_hot = pd.get_dummies(df['gender'])
    df = df.drop('gender',axis = 1)
    df = df.join(one_hot)

    one_hot = pd.get_dummies(df['maritalstatus'])
    df = df.drop('maritalstatus',axis = 1)
    df = df.join(one_hot)

    one_hot = pd.get_dummies(df['state'])
    df = df.drop('state',axis = 1)
    df = df.join(one_hot)
    return df

# Processing prediction probabilities and generating top 3 predictions
def prediction_generator(df_prediction, label_enocoder, prediction_matrix):
    top_3_index_matrix = []
    for element in range(len(prediction_matrix)):
        temp_index_holder = []
        top_3_idx = np.argsort(prediction_matrix[element])[-3:]
        temp_index_holder = list(top_3_idx)
        top_3_index_matrix.append(temp_index_holder)


    df_prediction['Reco_1_label'],df_prediction['Reco_2_label'],df_prediction['Reco_3_label'] = np.array(tt).T

    df_prediction['Reco_1'] = label_enocoder.inverse_transform(df_prediction['Reco_1_label'].values)
    df_prediction['Reco_2'] = label_enocoder.inverse_transform(df_prediction['Reco_2_label'].values)
    df_prediction['Reco_3'] = label_enocoder.inverse_transform(df_prediction['Reco_3_label'].values)

    del df_prediction['Reco_1_label']
    del df_prediction['Reco_2_label']
    del df_prediction['Reco_3_label']
    return df_prediction

# Reading processed csv file after preprocessing dump
df = pd.read('led_sql_process.csv')

# Dropping Nan's
df = df.dropna()

#Breaking DF into 2 parts
df_popular = pd.DataFrame()
df_popular = df[df['popular'] == 1]
df_non_popular = pd.DataFrame()
df_non_popular = df[df['popular'] == 0]

# Removing popular column from dataframe
del df_non_popular['popular']
del df_popular['popular']

# Removing customer_id column from dataframe
df_popular_preds = pd.DataFrame()
df_popular_preds['customer_id'] = df_popular['customer_id']
del df_popular['customer_id']

df_non_popular_preds = pd.DataFrame()
df_non_popular_preds['customer_id'] = df_non_popular['customer_id']
del df_non_popular['customer_id']

#One hot Encoding


#rearranging columns for preds
column_order = ['age', 'age_bin', 'yyyymm', 'yyyy', 'mm', 'Not Categorized', 'OTHERS', 'salaried', 'self_employed', 'Female', 'Male', 'DIVORCEE', 'MARRIED', 'UNMARRIED', 'WIDOW', 'AP', 'BH', 'BR', 'DELHI', 'GJ', 'HR', 'KA', 'KE', 'MH', 'MP', 'OR', 'PU', 'RJ', 'TN', 'UK', 'UP', 'WB']
df_non_popular = df_non_popular[column_order]
df_popular = df_popular[column_order]

# loading models
with open('R20.7/R20.7.model','rb') as f:
    model_popular = pickle.load(f)

with open('R20.15_rare/R20.15_rare.model','rb') as f:
    model_non_popular = pickle.load(f)

# loading label encoders
with open('R20.7/R20.7.le','rb') as f:
    le_popular = pickle.load(f)

with open('R20.15_rare/R20.15_rare.le','rb') as f:
    le_non_popular = pickle.load(f)

# converting dataframes to 2d matrix
df_popular_matrix = df_popular.values
df_non_popular_matrix = df_non_popular.values

# Getting predictions from model
popular_predicitons_matrix = model_popular.predict_proba(df_popular_matrix)
non_popular_predicitons_matrix = model_non_popular.predict_proba(df_non_popular_matrix)

# Processing the model predictions and generating top 3 recommendation
df_popular_preds = prediction_generator(df_popular_preds,le_popular,popular_predicitons_matrix)
df_non_popular_preds = prediction_generator(df_non_popular_preds,le_non_popular,non_popular_predicitons_matrix)

#Writing the recommendations to csv file with ',' as seperator
df_popular_preds.to_csv('popular_predictions_top3.csv',index=False)
df_non_popular_preds.to_csv('non_popular_predictions_top3.csv',index=None)
