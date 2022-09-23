import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import matplotlib.pyplot as plt
# import numpy as np
import plotly.express as px
from st_aggrid import GridOptionsBuilder, AgGrid
from style import add_logo

hide_menu_style = """
<style>
#MainMenu{visibility: hidden;}
footer{visibility:hidden;}
</style>
"""
# page expands to full width
st.set_page_config(page_title="Predicta.oil | Explore",
                   layout='wide', page_icon="⛽")
st.markdown(hide_menu_style, unsafe_allow_html=True)
st.title("Explore Models")
add_logo()
# ARIMA
# slider interval
interv = st.select_slider('Select Time Series Data Interval for Prediction', options=[
                          'Daily', 'Weekly', 'Monthly', 'Quarterly'], value='Weekly')

# dropdown 50 60 80
st.write("Select Split")
intervals = st.selectbox(
    "Select Interval:", ('80', '60', '50'))

# read file from select interval and dropdown split


def get_location(interv, intervals):
    # location = 'Explore/ARIMA/' + interv + '/' + intervals + '.csv'
    location = 'Explore/PREDICTIONS/' + interv + '/' + intervals + '.csv'
    return location


location = get_location(interv, intervals)

# pagination function for aggrid


def pagination(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True)
    return gb.build()


# read file
file = pd.read_csv(get_location(interv, intervals))
page = pagination(file)
file.drop("Unnamed: 0", axis=1, inplace=True)

# select columns
columns = file.columns.to_list()
# st.write(columns)
selectedCols = st.multiselect("Select models", columns, default=[
                              "Date", "Close Prices"])
df = file[selectedCols]
st.dataframe(df)

# print(df.columns.values)


fig = go.Figure()
for idx, col in enumerate(df.columns, 0):
    # print(df.iloc[1:,idx])
    if col != 'Date':
        fig.add_trace(go.Scatter(
            x=file['Date'], y=df.iloc[1:, idx], mode='lines', name=col))


st.plotly_chart(fig, use_container_width=True)

# LSTM
