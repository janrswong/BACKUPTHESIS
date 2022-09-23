import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# import numpy as np
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid
import plotly.graph_objects as go
from style import add_logo

hide_menu_style = """
<style>
#MainMenu{visibility: hidden;}
footer{visibility:hidden;}
</style>
"""

# page expands to full width
st.set_page_config(page_title="Predicta.oil | Home",
                   layout='wide', page_icon="⛽")
st.markdown(hide_menu_style, unsafe_allow_html=True)
add_logo()
# PAGE LAYOUT
# heading
st.title("Crude Oil Benchmark Stock Price Prediction LSTM and ARIMA Models")

st.header("Raw Data")

# select time interval
interv = st.select_slider('Select Time Series Data Interval for Prediction', options=[
    'Daily', 'Weekly', 'Monthly', 'Quarterly'], value='Weekly')

# st.write(interv[0])

# Function to convert time series to interval


@st.cache(persist=True, allow_output_mutation=True)
def getInterval(argument):
    switcher = {
        "W": "1wk",
        "M": "1mo",
        "Q": "3mo",
        "D": "1d"
    }
    return switcher.get(argument, "1wk")


# show raw data
# st.header("Raw Data")
# using button
# if st.button('Press to see Brent Crude Oil Raw Data'):


df = yf.download('BZ=F', interval=getInterval(interv[0]), end="2022-06-30")

# st.dataframe(df.head())
df = df.reset_index()


def pagination(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    return gb.build()


# enable enterprise modules for trial only
# raw data
page = pagination(df)
# AgGrid(df, enable_enterprise_modules=True,
#        theme='streamlit', gridOptions=page, fit_columns_on_grid_load=True, key='data')
# st.dataframe(df, width=2000, height=600)
# st.write(df)
st.table(df.head())
# download full data


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


csv = convert_df(df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='Brent Oil Prices.csv',
    mime='text/csv',
)


st.header("Standard Deviation of Brent Crude Oil")
sd = pd.read_csv('StandardDeviation.csv')
sd.drop("Unnamed: 0", axis=1, inplace=True)
# sd = sd.reset_index()
AgGrid(sd, key='SD1', enable_enterprise_modules=True,
       fit_columns_on_grid_load=True, theme='streamlit')
st.write("Note: All entries end on 2022-6-30.")

sd = sd.pivot(index='Start Date', columns='Interval',
              values='Standard Deviation')
sd = sd.reset_index()
# table
# AgGrid(sd, key='SD', enable_enterprise_modules=True,
#        fit_columns_on_grid_load=True, domLayout='autoHeight', theme='streamlit')

# visualization
fig = px.line(sd, x=sd.index, y=['1d', '1wk', '1mo', '3mo'],
              title="STANDARD DEVIATION OF BRENT CRUDE OIL PRICES", width=1000)
st.plotly_chart(fig, use_container_width=True)


# accuracy metrics
st.header("Accuracy Metric Comparison")
intervals = st.selectbox(
    "Select Interval:", ('Weekly', 'Monthly', 'Quarterly', 'Daily'), key='metricKey')
with st.container():
    col1, col2 = st.columns(2)

# LSTM METRICS
# st.write("LSTM Metrics")


readfile = pd.read_csv('LSTM.csv')
# readfile = readfile[readfile['Interval'] == intervals.upper()]
readfile = readfile[readfile['Interval'] == st.session_state.metricKey.upper()]
# readfile[readfile['Interval'] == intervals.upper()]
# readfile = updatefile(readfile)
readfile.drop("Unnamed: 0", axis=1, inplace=True)
with col1:
    st.write("LSTM Metrics")
    AgGrid(readfile, key=st.session_state.metricKey, fit_columns_on_grid_load=True,
           enable_enterprise_modules=True, theme='streamlit')


# st.write(st.session_state.metricKey)

# ARIMA METRICS
# st.write("ARIMA Metrics")
# intervals = st.selectbox(
#     "Select Interval:", ('Weekly', 'Monthly', 'Quarterly', 'Daily'))

if intervals == 'Weekly':
    file = pd.read_csv('ARIMAMetrics/ARIMA-WEEKLY.csv')
    file.drop("Unnamed: 0", axis=1, inplace=True)
    page = pagination(file)
    with col2:
        st.write("ARIMA Metrics")
        AgGrid(file, width='100%', theme='streamlit', enable_enterprise_modules=True,
               fit_columns_on_grid_load=True, key='weeklyMetric', gridOptions=page)

elif intervals == 'Monthly':
    file = pd.read_csv('ARIMAMetrics/ARIMA-MONTHLY.csv')
    file.drop("Unnamed: 0", axis=1, inplace=True)
    page = pagination(file)
    with col2:
        st.write("ARIMA Metrics")
        AgGrid(file, key='monthlyMetric', fit_columns_on_grid_load=True,
               enable_enterprise_modules=True, theme='streamlit', gridOptions=page)

elif intervals == 'Quarterly':
    file = pd.read_csv('ARIMAMetrics/ARIMA-QUARTERLY.csv')
    file.drop("Unnamed: 0", axis=1, inplace=True)
    page = pagination(file)
    with col2:
        st.write("ARIMA Metrics")
        AgGrid(file, key='quarterlyMetric', fit_columns_on_grid_load=True,
               enable_enterprise_modules=True, theme='streamlit', gridOptions=page)

elif intervals == 'Daily':
    file = pd.read_csv('ARIMAMetrics/ARIMA-DAILY.csv')
    file.drop("Unnamed: 0", axis=1, inplace=True)
    page = pagination(file)
    with col2:
        st.write("ARIMA Metrics")
        AgGrid(file, key='dailyMetric', width='100%', fit_columns_on_grid_load=True,
               enable_enterprise_modules=True, theme='streamlit', gridOptions=page)

# MODEL OUTPUT TABLE
st.header("Model Output (Close Prices vs. Predicted Prices)")

interval = st.selectbox("Select Interval:", ('Weekly',
                        'Monthly', 'Quarterly', 'Daily'), key='bestmodels')

if interval == 'Weekly':
    file = pd.read_csv('bestWeekly.csv')
    page = pagination(file)
    AgGrid(file, key='weeklycombined', fit_columns_on_grid_load=True,
           enable_enterprise_modules=True, theme='streamlit', gridOptions=page)

    # Visualization
    st.header("Visualization")
    fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(1, 0, 0)_Predictions",
                  "LSTM_80.0_Predictions"], title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)


elif interval == 'Monthly':
    file = pd.read_csv('bestMonthly.csv')
    page = pagination(file)
    AgGrid(file, key='monthlyCombined', fit_columns_on_grid_load=True,
           enable_enterprise_modules=True, theme='streamlit', gridOptions=page)
    # Visualization
    st.header("Visualization")
    fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_60.0_(0, 1, 1)_Predictions",  # find file
                  "LSTM_80.0_Predictions"], title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)


elif interval == 'Quarterly':
    file = pd.read_csv('bestQuarterly.csv')
    page = pagination(file)
    AgGrid(file, key='quarterlyCombined', fit_columns_on_grid_load=True,
           enable_enterprise_modules=True, theme='streamlit', gridOptions=page)
    # Visualization
    st.header("Visualization")
    fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(0, 1, 0)_Predictions",  # find file
                  "LSTM_80.0_Predictions"], title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)


elif interval == 'Daily':
    file = pd.read_csv('bestDaily.csv')
    page = pagination(file)
    AgGrid(file, key='dailyCombined', fit_columns_on_grid_load=True,
           enable_enterprise_modules=True, theme='streamlit', gridOptions=page)
    # Visualization
    st.header("Visualization")
    fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(0, 1, 0)_Predictions",  # find file
                                           "LSTM_60.0_Predictions"], title="BOTH PREDICTED BRENT CRUDE OIL PRICES", width=1000)
    st.plotly_chart(fig, use_container_width=True)
