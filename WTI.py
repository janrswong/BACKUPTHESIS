import matplotlib as mpl
from enum import auto
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# import numpy as np
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid
import plotly.graph_objects as go
from PIL import Image
import numpy as np


def displayWTI():
    st.header("Raw Data")
    # select time interval
    interv = st.select_slider('Select Time Series Data Interval for Prediction', options=[
        'Daily', 'Weekly', 'Monthly'], value='Weekly')
    # st.write(interv[0])
    # Function to convert time series to interval

    @st.cache(persist=True, allow_output_mutation=True)
    def getInterval(argument):
        switcher = {
            "W": "WTI/Weekly-WTI.csv",
            "M": "WTI/Monthly-WTI.csv",
            "D": "WTI/Daily-WTI.csv"
        }
        return switcher.get(argument, "WTI/Weekly-WTI.csv")

    df = pd.read_csv(getInterval(interv[0]))

    def pagination(df):
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_pagination(paginationAutoPageSize=True)
        return gb.build()

    page = pagination(df)
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
        file_name='WTI Oil Prices.csv',
        mime='text/csv',
    )

    # st.header("Standard Deviation of Raw Data")
    # sd = pd.read_csv('StandardDeviation.csv')
    # sd.drop("Unnamed: 0", axis=1, inplace=True)
    # # sd = sd.reset_index()
    # AgGrid(sd, key='SD1', enable_enterprise_modules=True,
    #        fit_columns_on_grid_load=True, theme='streamlit')
    # st.write("Note: All entries end on 2022-06-30.")

    # sd = sd.pivot(index='Start Date', columns='Interval',
    #               values='Standard Deviation')
    # sd = sd.reset_index()
    # # visualization
    # fig = px.line(sd, x=sd.index, y=['1d', '1wk', '1mo', '3mo'],
    #               title="STANDARD DEVIATION OF BRENT CRUDE OIL PRICES", width=1000)
    # st.plotly_chart(fig, use_container_width=True)

    # accuracy metrics
    st.header("Accuracy Metric Comparison")
    intervals = st.selectbox(
        "Select Interval:", ('Daily', 'Weekly', 'Monthly'), key='metricKey')
    with st.container():
        col1, col2 = st.columns(2)

    # LSTM METRICS
    # st.write("LSTM Metrics")

    readfile = pd.read_csv('WTI/LSTM.csv')
    # readfile = readfile[readfile['Interval'] == intervals.upper()]
    readfile = readfile[readfile['Interval']
                        == st.session_state.metricKey.upper()]
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
        file = pd.read_csv('WTI/ARIMAMetrics/ARIMA-WEEKLY.csv')
        file.drop("Unnamed: 0", axis=1, inplace=True)
        page = pagination(file)
        with col2:
            st.write("ARIMA Metrics")
            AgGrid(file, width='100%', theme='streamlit', enable_enterprise_modules=True,
                   fit_columns_on_grid_load=True, key='weeklyMetric', gridOptions=page)

    elif intervals == 'Monthly':
        file = pd.read_csv('WTI/ARIMAMetrics/ARIMA-MONTHLY.csv')
        file.drop("Unnamed: 0", axis=1, inplace=True)
        page = pagination(file)
        with col2:
            st.write("ARIMA Metrics")
            AgGrid(file, key='monthlyMetric', fit_columns_on_grid_load=True,
                   enable_enterprise_modules=True, theme='streamlit', gridOptions=page)

    elif intervals == 'Daily':
        file = pd.read_csv('WTI/ARIMAMetrics/ARIMA-DAILY.csv')
        file.drop("Unnamed: 0", axis=1, inplace=True)
        page = pagination(file)
        with col2:
            st.write("ARIMA Metrics")
            AgGrid(file, key='dailyMetric', width='100%', fit_columns_on_grid_load=True,
                   enable_enterprise_modules=True, theme='streamlit', gridOptions=page)

    # TABLES
    # BRENT WTI
    st.header("Brent vs. WTI Comparison")
    st.subheader('ARIMA Accuracy Metrics & Best Models')
    df2 = pd.DataFrame([[0.8, (0, 1, 0), 2.427, 0.017, 0.8, (0, 1, 0), 5.211, 0.023], [np.nan, np.nan, np.nan, np.nan, 0.5, (0, 1, 0), 9.498, 0.042], [0.5, (1, 0, 0), 9.366, 0.039, 0.500000, (1, 0, 0), 9.530000, 0.042000], [np.nan, np.nan, np.nan, np.nan, 0.500000, (0, 1, 0), 41.668000, 0.097000], [0.600000, (0, 1, 1), 46.308000, 0.091000, 0.600000, (0, 1, 1), 45.242000, 0.099000]], index=pd.Index(
        ['Daily', 'Weekly*', 'Weekly', 'Monthly*', "Monthly"], name='Actual Label:'),
        columns=(["Brent Train Split", "Brent Order", "Brent MSE", "Brent MAPE", "WTI Train Split", "WTI Order", "WTI MSE", "WTI MAPE"]))

    cell_hover = {  # for row hover use <tr> instead of <td>
        'selector': 'tr:hover',
        'props': [('background-color', '#ff4c4c')]
    }

    index_names = {
        'selector': '.index_name',
        'props': 'font-style: italic; color: darkgrey; font-weight:normal;'
    }
    headers = {
        'selector': 'th:not(.index_name)',
        'props': 'background-color: #f0f2f6; color: black;'
    }
    df2 = df2.style
    df2 = df2.set_table_styles(
        [cell_hover, index_names, headers]).highlight_null(props="color: transparent;")
    st.table(df2)

    # LSTM
    st.subheader('LSTM Accuracy Metrics & Best Models')
    df3 = pd.DataFrame([[0.6, 4.152, 0.021, 0.8, 5.904, 0.02], [0.8, 21.62, 0.037, 0.8, 25.012, 0.039], [0.8, 56.275, 0.075, 0.8, 80.147, 0.096]], index=pd.Index(
        ['Daily', 'Weekly', "Monthly"], name='Actual Label:'),
        columns=(["Brent Train Split", "Brent MSE", "Brent MAPE", "WTI Train Split", "WTI MSE", "WTI MAPE"]))
    df3 = df3.style
    df3 = df3.set_table_styles(
        [cell_hover, index_names, headers])
    st.table(df3)

    # MODEL OUTPUT TABLE
    st.header("Model Output (Close Prices vs. Predicted Prices)")

    interval = st.selectbox("Select Interval:", ('Daily', 'Weekly',
                                                 'Monthly'), key='bestmodels')

    if interval == 'Weekly':
        file = pd.read_csv('WTI/BestWTI/bestWeekly.csv')
        page = pagination(file)
        AgGrid(file, key='weeklycombined', fit_columns_on_grid_load=True,
               enable_enterprise_modules=True, theme='streamlit', gridOptions=page)

        # Visualization
        st.header("Visualization")
        fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(0, 1, 0)_Predictions",
                                               "ARIMA_50.0_(1, 0, 0)_Predictions", "LSTM_80.0_Predictions"], title="BOTH PREDICTED WTI CRUDE OIL PRICES", width=1000)
        st.plotly_chart(fig, use_container_width=True)

    elif interval == 'Monthly':
        file = pd.read_csv('WTI/BestWTI/bestMonthly.csv')
        page = pagination(file)
        AgGrid(file, key='monthlyCombined', fit_columns_on_grid_load=True,
               enable_enterprise_modules=True, theme='streamlit', gridOptions=page)
        # Visualization
        st.header("Visualization")
        fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_50.0_(0, 1, 0)_Predictions",
                                               "ARIMA_60.0_(0, 1, 1)_Predictions", "LSTM_80.0_Predictions"], title="BOTH PREDICTED WTI CRUDE OIL PRICES", width=1000)
        st.plotly_chart(fig, use_container_width=True)

    elif interval == 'Daily':
        file = pd.read_csv('WTI/BestWTI/bestDaily.csv')
        page = pagination(file)
        AgGrid(file, key='dailyCombined', fit_columns_on_grid_load=True,
               enable_enterprise_modules=True, theme='streamlit', gridOptions=page)
        # Visualization
        st.header("Visualization")
        fig = px.line(file, x=file["Date"], y=["Close Prices", "ARIMA_80.0_(0, 1, 0)_Predictions",  # find file
                                               "LSTM_60.0_DAILY", "LSTM_80.0_DAILY", ], title="BOTH PREDICTED WTI CRUDE OIL PRICES", width=1000)
        st.plotly_chart(fig, use_container_width=True)
