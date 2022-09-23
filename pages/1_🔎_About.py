import streamlit as st
from PIL import Image
import base64
from style import add_logo


def gif(location):
    """### gif from local file"""
    file_ = open(location, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

#     return (st.markdown(
#         f'<img src="data:image/gif;base64,{data_url}" alt="instructions gif">',
#         unsafe_allow_html=True,
#     ))
    return (st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="instructions gif" width="100%">',
        unsafe_allow_html=True,
    ))


hide_menu_style = """
<style>
#MainMenu{visibility: hidden;}
footer{visibility:hidden;}
</style>
"""
# page expands to full width
st.set_page_config(page_title="Predicta.oil | About",
                   layout='wide', page_icon="⛽")
st.markdown(hide_menu_style, unsafe_allow_html=True)
add_logo()
st.title('About the Research')
# a. how the thing works
st.header('How It Works')
st.markdown('<strong>Welcome!</strong> This is a fully interactive, multi-page web app through the Python library Streamlit that allows users to explore the same models used in the study. Aside from learning about study findings, play with parameters, create your own models, conduct your own comparisions and make your own analyses! Read further to learn how to use the <em>Explore</em> and <em>Make Your Own Model</em> tabs.', unsafe_allow_html=True)

# a.1 How to use the explore tab
st.markdown('### How to use the <a href="/Explore" target="_self" style="text-decoration:none; color: tomato;">📈 Explore</a>' ' tab', unsafe_allow_html=True)
# st.subheader("How to use the 'Explore' tab")
st.markdown('<p> The Explore tab allows you to <strong>select different models</strong> based on time intervals and test-train split data. <br> </p>', unsafe_allow_html=True)

st.markdown(
    '<p> Use the slider to select whether or not you would like to see monthly, weekly, daily, or quarterly data being used. </p>', unsafe_allow_html=True)
gif('assets/images/Slider1.gif')


st.markdown('<p> Then, use the dropdown to select your split. </p>',
            unsafe_allow_html=True)
gif('assets/images/Split1.gif')
# split = Image.open('assets/images/Split.jpg')
# st.image(split, caption='Explore Split')

st.markdown('<p>  Afterwards, you can use the select tool to compare as many ARIMA models as you like, either against each other or against an LSTM model! </p>', unsafe_allow_html=True)
# select = Image.open('assets/images/Select.jpg')
# st.image(select, caption='Explore Select')
gif('assets/images/Select1.gif')


# a.2 How to use the Make a model tab
st.markdown('### How to use the <a href="Make_a_Model" target="_self" style="text-decoration:none; color: tomato;">📊 Make a Model</a>' ' tab', unsafe_allow_html=True)
st.markdown('<p> The Make a Model tab lets you <strong>create your own model</strong>! Choose the different parameters and test-train splits to your liking for an interactive experience with our time series models. <br> </p>', unsafe_allow_html=True)

st.markdown('<p> Use the slider to select your preferred time interval. You can also download the full raw data in CSV format by clicking the button below! </p>', unsafe_allow_html=True)
gif('assets/images/Slider1.gif')
gif('assets/images/Download_CSV.gif')

st.markdown('<p> Get acquanited with all the controls on the page by starting with the side menu. Here, you can adjust the train-test split and ARIMA p, d, and q values as you like! </p>', unsafe_allow_html=True)
# gif('assets/images/Set_pdq_values.gif')

st.markdown('<p> Check the "Details" box to expand the page for more information, and press the button to generate your model! </p>', unsafe_allow_html=True)
gif('assets/images/pdq.gif')

st.markdown('<p> You can view how your model fares against the original close prices on the chart, and even see how the model you made stacks up to accuracy metrics! </p>', unsafe_allow_html=True)
accuracy = Image.open('assets/images/make.png')
st.image(accuracy, caption='Visualization')

# b. snippets of the paper

st.markdown('<h1>Conclusions and Recommendations</h1> <p>The full documentation of this project can be accessed through this link: [https://bit.ly/PredictaPaper] </p><h2>Conclusions</h2> <h3>Price Movement Volatility Trends</h3> <p> Price movement volatility refers to how much a set of prices changes over time and how erratic those changes are. In crude oil prices, unless there are spikes or drops due to unforeseen or anomalous circumstances, these trends tend to stray away from erratic highs and lows especially over short periods of time. It must be reiterated that this study does not take into account these anomalies, but focuses on what would be the natural, steady trend of Brent crude oil prices. That being said, the conduction of the study simply paints a clear picture of the behavior of asset prices and how the value of volatility changes over spans of time. </p>  <p> In order to quantify volatility, the standard deviation between the actual close prices (prices from the yfinance dataset) and the predicted prices are computed. From this part of the experiment, it was found that volatility is more likely to be present over longer periods of time. Additionally, it can be observed that volatility is also contingent on anomalous or external factors.</p> <p>It was found that Brent crude oil in particular shows the highest volatility trend over quarterly prices.</p>', unsafe_allow_html=True)

st.markdown('<h3>Model Accuracy</h3> <p> Model accuracy is quantified through the use of accuracy metrics, specifically MAPE and MSE. These subsections are partitioned in accordance to the time interval of the raw data used, that being daily, weekly, monthly, and quarterly close prices.</p> <h4>Daily Interval Data</h4> <p>It was found that 96 of 102 or 94.12% ARIMA models and 3 of 3 or 100.00% LSTM models were able to attain a MAPE percentage below 10% and 96 of 102 or 94.12% ARIMA models and 3 of 3 or 100.00% LSTM models were able to attain a MSE percentage close to 0 or less than 0.1 using daily interval data.</p> <h4>Weekly Interval Data</h4> <p>It was found that 42 of 48 or 87.50% ARIMA models and 0 of 3 or 0.00% LSTM models were able to attain a MAPE percentage below 10% and 22 of 48 or 45.83%  ARIMA models and 0 of 3 or 0.00% LSTM models were able to attain a MSE percentage close to 0 or less than 0.1 using weekly interval data.</p> <h4>Monthly Interval Data</h4> <p>It was found that 62 of 160 or 38.75% ARIMA model and 1 of 3 or 33.33% LSTM models were able to attain a MAPE percentage below 10% and 0 of 160 or 0.00% ARIMA models and 0 of 3 or 0.00% LSTM models were able to attain a MSE percentage close to 0 or less than 0.1 using monthly interval data. </p> <h4>Quarterly Interval Data</h4> <p>It was found that 0 of 77 or 0.00% ARIMA models and 0 of 3 or 0.00% LSTM models were able to attain a MAPE percentage below 10% and 0 of 77 or 0.00%  ARIMA models and 0 of 3 or 0.00% LSTM models were able to attain a MSE percentage close to 0 or less than 0.1 using quarterly interval data.</p>', unsafe_allow_html=True)

st.markdown('<h3>Conclusion</h3> <p>It must be noted that despite most models falling short of the standards established, these standards are the most ideal values for MSE and MAPE that indicate the near-perfectness of a model. It must also be noted that the use of parameter grid search for the p, d, and q values of ARIMA effect the amounts of total ARIMA models. </p> <p> When presented with the entirety of both cumulative ARIMA and LSTM predictions against the entirely of close prices, it can be observed that closer intervals like daily and weekly render closer predictions than that of monthly and quarterly predictions. This can be attributed to the fact that within a pool of raw, close price data from 2007 to 2022, there is more data to work with given daily prices, 365 days a year, as opposed to monthly or quarterly prices that only account for 12 months and 3 months a year, respectively. This reflects in the aforementioned results of the MSE and MAPE, whose results act as supplementary metrics for accuracy comparison. </p>', unsafe_allow_html=True)

st.markdown('<h3>Model Comparison</h3> <p>Between the two models, the most accurate one will have the lowest percentage of models that have a MAPE percentage close to or less than 10\%, and MSE percentage close to 0, and the quickest runtimes. The quantification of performance is very reliant on the runtime as well. </p> <p> In terms of MSE accuracy and performance, it was found that ARIMA produced the highest percentage of most accurate models consistently throughout the four time intervals, in comparison to LSTM. On the other hand, in terms of MAPE accuracy and performance, ARIMA still proved to produce the highest percentage of outcomes over the time intervals as opposed to LSTM. The best ARIMA model rendered was through a 50% test-train split, completed with a run time of 18.755 milliseconds and parameters of (0,1,0).</p> <p> For daily close prices, ARIMA, with 94.12% of models accurate, fell short to LSTM which had 100.00% models accurate for MAPE. MSE rendered the same results for both model types. Weekly close prices had ARIMA boasting a 87.50% percentage in MAPE while LSTM rendered 0.00%. ARIMA scored a 45.83% for MSE while LSTM retained the same value of 0.00%. Monthly interval data returned 38.75% of ARIMA models and 33.33% of LSTM models for MAPE. MSE sae 0.00% for both ARIMA and LSTM. Lastly, quarterly close prices returned 0.00% for both models across both LSTM and ARIMA. </p> <p> From this comparison, it can be concluded that ARIMA is more consistent in having better accuracy than LSTM through these accuracy metrics. It can be also inferred that both LSTM and ARIMA fare better given closer time intervals, and in turn when there is more raw data to work with.</p> While the overall proximity to accuracy of either model was discussed, it can be noted that that there are model results that stand out from each model type. This is important as even though the best model might be from a certain model type, it does not effect the overall performance of the model. It can, however, illustrate how well either model can process data over specific intervals of time.<p> For reference, the best models of ARIMA and LSTM are presented. The best daily ARIMA model renders an MSE of 2.427 and a MAPE of 0.017 while the LSTM model renders 4.152 for MSE and 0.021 for MAPE; making ARIMA the best for daily. For weekly, ARIMA still performs better. However a shift becomes apparent for the best MAPE models for monthly and quarterly intervals. In terms of splits, 50 splits for ARIMA and 80 splits for LSTM made up most of the best models.</p><p> </p><p> In conclusion, the ARIMA model proves itself to be more accurate and more efficient. However, if individual accuracy metric outcomes are to be considered, the best metric for ARIMA is MSE and MAPE for daily, weekly, monthly, and quarterly prices, while the best metric for LSTM is MAPE for long term intervals like monthly and quarterly. In terms of split, ARIME needed less training time while LSTM required more training. </p>', unsafe_allow_html=True)

st.markdown('<h2>Recommendations</h2> <p>With the utilization of Google Colab in implementing the LSTM and ARIMA models, its collaboration-friendly environment and dedicated cloud storage service are good features for this study. However, the notebook has a disadvantage of having only a maximum of 12 hours runtime for each execution which is below the runtime of ARIMA models for the daily intervals, thus resulting to cutting off outputs. With this, the researchers recommend to run the models in local IDEs and saving the outputs in CSV to local storage as well. Moreover, for a more organized and secured compilation of output data, connection to a database is highly recommended to the future researchers adapting this study.</p> <p>Furthermore, the researchers observed the great amount of time needed to be invested in running the models in the Google Colab due to its heavy load of data available from 2007. Hence, it is recommended to allot ample time for more variations of testing and training to further evaluate the best model/s when running models with different intervals, train-test split, and parameters. In addition, the researchers have initially assessed through previous studies that MSE and MAPE are the two most common accuracy metrics used for time series data. The researchers further suggest to expand the options for accuracy metrics in order to develop more non-biased evaluations on the best models for specified time intervals and have a more detailed comparison between accuracy metrics.</p> <p>To further relate the accuracy of the models, it is also recommended to also focus on adding future predictions of the Brent Crude Oil Prices. These will be a significant information for stakeholders especially with the constant shifts of economy disposition in the country. Lastly, as much as Brent Crude Oil is a relevant commodity to focus on this kind of prediction study with its massive increase in price in the past few months, the researchers also encourage to try the LSTM and ARIMA models to other valuable stocks and assets available in the industry. The researchers have discovered the importance of Philippine economy-related studies most especially to investors, economics professionals, and policymakers in taking a course of action towards economic progression.</p>', unsafe_allow_html=True)


# # sample text
# st.markdown('Here are some code snippets!')
# st.markdown('Here are some code snippets that illustrate the saving and evaluation of the ARIMA and LSTM models in Google Colab')

# # sample add code snippet

# code = '''# evaluate combinations of p, d and q values for an ARIMA model
# # prepare training dataset
# train_size = int(len(X) * split)
# train, test = X[0:train_size], X[train_size:]
# history = [x for x in train]
# predset = pd.DataFrame(test)

# # make predictions
# predictions = list()
# for t in range(len(test)):
#      model = ARIMA(history, order=arima_order)
#      model_fit = model.fit()
#      yhat = model_fit.forecast()[0]
#      predictions.append(*yhat)
#      history.append(test[t])'''
# st.code(code, language='python')
# code = '''p_values = range(0, 3)
# d_values = range(0, 3)
# q_values = range(0, 2)
# split_values = [0.8, 0.5, 0.6]
# warnings.filterwarnings("ignore")
# df1 = evaluate_models(df.Close, p_values, d_values, q_values,split_values)'''
# st.code(code, language='python')
# # LSTM
# code = '''# LSTM Model
# # preprocessing
#     date_train, date_val, date_test = df.index[:int(df.shape[0]*split)],df.index[int(df.shape[0]*split):int(df.shape[0]*val_split)], df.index[int(df.shape[0]*val_split)+WINDOW_SIZE:]
#     X_train1, y_train1 = X1[:int(df.shape[0]*split)], y1[:int(df.shape[0]*split)]
#     X_val, y_val = X1[int(df.shape[0]*split):int(df.shape[0]*val_split)], y1[int(df.shape[0]*split):int(df.shape[0]*val_split)]
#     X_test1, y_test1 = X1[int(df.shape[0]*val_split):], y1[int(df.shape[0]*val_split):]

#     # X_train1.shape, y_train1.shape, X_test1.shape, y_test1.shape

#     # lstm model
#     model = Sequential([layers.Input((3,1)),layers.LSTM(64),layers.Dense(32, activation='relu'),layers.Dense(32, activation='relu'), layers.Dense(1)])
#     cp1 = ModelCheckpoint('model1/', save_best_only=True)
#     model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mean_absolute_percentage_error'])
#     model.fit(X_train1, y_train1,validation_data=(X_val, y_val), epochs=100, callbacks=[cp1])
#     # model.summary()
# '''
# st.code(code, language='python')


# # code = '''def hello():
# #      print("Hello, Streamlit!")'''
# # st.code(code, language='python')

# # sample add latex or formulas
# st.latex(r'''
#      a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
#      \sum_{k=0}^{n-1} ar^k =
#      a \left(\frac{1-r^{n}}{1-r}\right)
#      ''')

# c. about us creators
st.header('About the Authors')
st.markdown('The authors, Janna Rizza Wong, Stephanie Lorraine Ignas, and Alyanna Angela Castillon, are all university seniors taking Bachelor of Science in Computer Science in Ateneo de Davao University, Philippines.')
