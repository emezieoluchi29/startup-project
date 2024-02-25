import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import streamlit as st
import joblib

data = pd.read_csv('startUp(2).csv')
# model = joblib.load('startUpModel.pkl')

df = data.copy()

# state(text) is not linearly related to the profit(number)
df.drop(['State', 'Unnamed: 0'], axis = 1, inplace = True) # ..... Drop state becuase linear regression dont take text

# Standard Scaling the columns individually
from sklearn.preprocessing import StandardScaler

# rd_spend
rd_spend_scale = StandardScaler()
df['R&D Spend'] = rd_spend_scale.fit_transform(df[['R&D Spend']])

# Mgt 
mgt_scale = StandardScaler()
df['Administration'] = mgt_scale.fit_transform(df[['Administration']])

# Marketting 
mkt_scale = StandardScaler()
df['Marketing Spend'] = mkt_scale.fit_transform(df[['Marketing Spend']])


# Train and Test 
from sklearn.model_selection import train_test_split
x = df.drop(['Profit'], axis = 1)
y = df.Profit

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 7)
print(f'xtrain: {xtrain.shape}')
print(f'xtest: {xtest.shape}')
print('ytrain: {}'.format(ytrain.shape))
print('ytest: {}'.format(ytest.shape))

# MODELLING ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

lin_reg = LinearRegression()
lin_reg.fit(xtrain, ytrain)


# -------------------------- STREAMLIT IMPLEMENTATION ------------------------------
st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: helvetica'>STARTUP PROFIT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By PluralCode Data Science Cohort</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com.png')

st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>Project Overview</h4>", unsafe_allow_html = True)

st.write("The goal of this project is to develop a predictive model that assesses the profitability of startup companies. By leveraging machine learning techniques, we aim to provide insights into the factors influencing a startup's financial success, empowering stakeholders to make informed decisions")

st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data, use_container_width= True)

st.sidebar.image('pngwing.com.png', caption = 'Welcome Dear User')

rd_spend = st.sidebar.number_input('Research and Development')
admin = st.sidebar.number_input('Administration Expense',)
mkt_exp = st.sidebar.number_input('Marketing Expense')

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>Input Variable</h4>", unsafe_allow_html = True)

inputs = pd.DataFrame()
inputs['R&D Spend'] = [rd_spend]
inputs['Administration'] = [admin]
inputs['Marketing Spend'] = [mkt_exp]

st.dataframe(inputs, use_container_width= True)

#transforming
inputs['R&D Spend'] = rd_spend_scale.transform(inputs[['R&D Spend']])
inputs['Administration'] = mgt_scale.transform(inputs[['Administration']])
inputs['Marketing Spend'] = mkt_scale.transform(inputs[['Marketing Spend']])


prediction_button=st.button('Predict Profitability')
if prediction_button:
    Predicted = lin_reg.predict(inputs)
    st.success(f'The profit perdicted for your company is {Predicted[0].round(2)}')

# # Model Prediction 
# predicted = model.predict(inputs)
# st.success(predicted.round(2))