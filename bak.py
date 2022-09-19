from statsmodels.tsa.arima_model import ARIMAResults
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import joblib
import math
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# page expands to full width
st.set_page_config(page_title="LSTM vs ARIMA", layout='wide')

# PAGE LAYOUT
# heading
st.title("Crude Oil Benchmark Stock Price Prediction LSTM and ARIMA Models")
st.subheader("""© Castillon, Ignas, Wong""")

# ARIMA PARAMETERS
pValue = 4
dValue = 1
qValue = 0


# sidebar
# Sidebar - Specify parameter settings
with st.sidebar.header('Set Data Split'):
  # PARAMETERS min,max,default,skip
    trainData = st.sidebar.slider(
        'Data split ratio (% for Training Set)', 10, 90, 80, 5)
    # st.write(trainData*.01)
    accuracy = st.sidebar.select_slider(
        'Performance measure (accuracy Metrics)', options=['both', 'mse', 'mape'])
    # ARIMA PARAMETERS
    pValue = st.sidebar.number_input('P-value:', 0, 100, pValue)
    st.sidebar.write('The current p-Value is ', pValue)
    dValue = st.sidebar.number_input('D-value:', 0, 100, dValue)
    st.sidebar.write('The current d-Value is ', dValue)
    qValue = st.sidebar.number_input('Q-value:', 0, 100, qValue)
    st.sidebar.write('The current q-Value is ', qValue)

# download

# model selection
modSelect = st.selectbox("Select Model for Prediction:",
                         ("ARIMA & LSTM", "LSTM", "ARIMA"))

# //show option selected
# st.write(modSelect)

# select time interval
interv = st.select_slider('Select Time Series Data Interval for Prediction', options=[
                          'Weekly', 'Monthly', 'Quarterly', 'Yearly'])

# st.write(interv[0])

# Function to convert time series to interval


def getInterval(argument):
    switcher = {
        "W": "1wk",
        "M": "1mo",
        "Q": "3mo",
        "Y": "1d"
    }
    return switcher.get(argument, "1d")


# show raw data
st.header("Raw Data")
# using button
# if st.button('Press to see Brent Crude Oil Raw Data'):
df = yf.download('BZ=F', interval=getInterval(interv[0]))
df

# graph visualization
st.header("Visualizations")
# LSTM


def df_to_X_y(df, window_size=5):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size]
        y.append(label)
    return np.array(X), np.array(y)


def mse_eval(test, predictions):
    return mean_squared_error(test, predictions)


def mape_eval(test, predictions):
    return mean_absolute_percentage_error(test, predictions)


def evaluate_lstm_model(split):
    WINDOW_SIZE = 3
    X1, y1 = df_to_X_y(df['Close'], WINDOW_SIZE)

    # preprocessing
    date_train, date_test = df.index[:int(
        df.shape[0]*split)], df.index[int(df.shape[0]*split)+WINDOW_SIZE:]
    X_train1, y_train1 = X1[:int(df.shape[0]*split)
                            ], y1[:int(df.shape[0]*split)]
    X_test1, y_test1 = X1[int(df.shape[0]*split):], y1[int(df.shape[0]*split):]
    # X_train1.shape, y_train1.shape, X_test1.shape, y_test1.shape

    # lstm model
    model = Sequential([layers.Input((3, 1)), layers.LSTM(64), layers.Dense(
        32, activation='relu'), layers.Dense(32, activation='relu'), layers.Dense(1)])
    cp1 = ModelCheckpoint('model1/', save_best_only=True)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001),
                  metrics=['mean_absolute_percentage_error'])
    model.fit(X_train1, y_train1, epochs=100, callbacks=[cp1])
    model.summary()
    # train predictions
    train_predictions = model.predict(X_train1).flatten()
    train_results = pd.DataFrame(
        data={'Date': date_train, 'Close Prices': y_train1, 'Train Predictions': train_predictions})
    # train_results

    # test predictions
    test_predictions = model.predict(X_test1).flatten()
    test_results = pd.DataFrame(
        data={'Date': date_test, 'Close Prices': y_test1, 'LSTM Predictions': test_predictions})
    # test_results

    # evaluate model
    mse = mse_eval(test_results['Close Prices'],
                   test_results['LSTM Predictions'])
    mape = mape_eval(test_results['Close Prices'],
                     test_results['LSTM Predictions'])
    print(mse)
    print(mape)
    # # save to csv
    # # csv file
    # current_name_model = str('LSTM_'+str(split*100))
    # predict = '/home/janna/1thesis/testingthesis/CSVPREDICTIONS_' + \
    #     current_name_model + '.csv'
    # test_results.to_csv(predict, float_format='%.2f')

    # plot orig price and predicted price
    fig = px.line(test_results, x=test_results['Date'], y=["Close Prices", "LSTM Predictions"],
                  title="LSTM PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)
    # VISUALIZE DATA
    plt.figure(figsize=(24, 24))
    plt.grid(True)
    return test_results


results = evaluate_lstm_model(trainData*.01)
results
# # model

# ARIMA MODEL
# TRAIN,TEST,&SPLIT DATA

# split data
row = int(len(df)*(trainData*.01))  # 80% testing

trainingData = list(df[0:row]['Close'])
# len(trainingData)
testingData = list(df[row:]['Close'])
# len(testingData)
# using historical data to predict future data

predictions = []
nObservations = len(testingData)

for i in range(nObservations):
    model = ARIMA(trainingData, order=(pValue, dValue, qValue))  # p,d,q
    # model = sm.tsa.arima.ARIMA(trainingData, order=(4,1,0)) #p,d,q
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = list(output[0])[0]
    predictions.append(yhat)
    actualTestValue = testingData[i]
    # update training set
    trainingData.append(actualTestValue)
    # print(output)
    # break

# print summary
details = st.checkbox('Details')

arimamodsum = model_fit.summary()
if details:
    st.write(arimamodsum)

# st.write(predictions)
predictionss = pd.DataFrame(predictions)
# df['ARIMApredictions'] = predictions

# df = pd.insert([predictionss])

# st.write(predictionss)
# df

testingSet = pd.DataFrame(testingData)
testingSet['ARIMApredictions'] = predictions
testingSet.columns = ['Close Prices', 'ARIMA Predictions']
testingSet

results["ARIMA Predictions"] = testingSet["ARIMA Predictions"]
results

# # plot orig price and predicted price
# fig = px.line(testingSet, x=testingSet.index, y=["Close Prices","ARIMA Predictions"],
#     title="ARIMA PREDICTED BRENT CRUDE OIL PRICES", width=1000)
# st.plotly_chart(fig, use_container_width=True)

# plot orig price and predicted price
fig = px.line(results, x=results["Date"], y=["Close Prices", "ARIMA Predictions", "LSTM Predictions"],
              title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
st.plotly_chart(fig, use_container_width=True)

# #VISUALIZE DATA
# plt.figure(figsize=(24,24))
# plt.grid(True)

# dateRange = df[row:].index

# plt.plot(dateRange, predictions, color='blue', marker = 'o', linestyle ='dashed', label='Predicted Brent Price')
# plt.plot(dateRange, testingData, color='red', label='Original Brent Price')

# plt.title(" ARIMA BRENT PRICE PREDICTION")
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

mape = np.mean(np.abs(np.array(predictions) -
               np.array(testingData))/np.abs(testingData))
mse = np.square(np.subtract(testingData, predictions)).mean()
MSE = mean_squared_error(testingData, predictions)
MAPE = mean_absolute_percentage_error(testingData, predictions)
MAE = mean_absolute_error(testingData, predictions)

st.write("MAPE: " + str(mape))  # Mean absolute Percentage Error
st.write("MAPE: " + str(MAPE))  # Mean absolute Percentage Error
st.write("MSE: " + str(mse))  # MSE
st.write("MSE: " + str(MSE))  # MSE

accTable = pd.DataFrame()
accTable['MAPE'] = [mape]
accTable['MSE'] = [mse]
accTable['Improved'] = [2200]

# accuracy metrics
st.header("Accuracy Metrics")

st.table(accTable)

# ______________________________________________________
# sample read from local file!!!
readfile = pd.read_csv('ARIMA/Sheets/ARIMA-WEEKLY.csv')
readfile

# load csv
# file = pd.read_csv('./PREDICTIONS_ARIMA_80.0.csv')
file = pd.read_csv('./PREDICTIONS_ARIMA_80.0_(4,2,2).csv')
file
# load model
# loaded = ARIMAResults.load('ARIMA_80.0.pkl')
loaded = ARIMAResults.load('ARIMA_80.0_(4, 2, 2).pkl')
st.write(loaded.summary())

# file['ARIMA Predictions']
# file['Close Prices']

# # evaluate model
# mse = float(mse_eval(file['Close Prices'],file['ARIMA Predictions']))
# mape = mape_eval(file['Close Prices'],file['ARIMA Predictions'])
# print("MSE: "+ str(mse))
# print("MAPE: "+ str(mape))
