import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="ArysStockAnalysis",
    page_icon="🧊",
    initial_sidebar_state="expanded",
)
 
st.title('arys stock analysis toolkit 🚀')
st.write("""
Hello 👋 and welcome to arys **stock analysis toolkit**.
It uses multiple **technical analytical** stratergies to screen and analyse stocks and market 💪
""")
st.link_button("GitHub", "https://github.com/aryaneelshivam/ArysStockAnalysis")
st.divider()

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
import streamlit as st
from datetime import date, timedelta
from tradingview_ta import TA_Handler, Interval, Exchange

# Set Matplotlib style
plt.style.use('fivethirtyeight')
yf.pdr_override()

# Sidebar for user input
st.sidebar.title("Stock Analysis")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "ZOMATO.NS")

# Date Range Selection
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", date.today())
st.sidebar.divider()
st.sidebar.write('arys is an analytical toolkit created by **Aryaneel Shivam** to screen stock prices with some **technical parameters**')


# Download stock data
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Calculate technical indicators
rstd = stock_data['Close'].rolling(window=15).std()
stock_data['EMA5'] = stock_data['Close'].ewm(span=5).mean()
stock_data['EMA15'] = stock_data['Close'].ewm(span=15).mean()
stock_data['SMA5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['SMA15'] = stock_data['Close'].rolling(window=15).mean()
upper_band = stock_data['SMA15'] + 2 * rstd
lower_band = stock_data['SMA15'] - 2 * rstd

# Buy and Sell signals for SMA
def buy_sell(stock_data):
    signalBuy = []
    signalSell = []
    position = False

    for i in range(len(stock_data)):
        if stock_data['SMA5'][i] > stock_data['SMA15'][i]:
            if not position:
                signalBuy.append(stock_data['Adj Close'][i])
                signalSell.append(np.nan)
                position = True
            else:
                signalBuy.append(np.nan)
                signalSell.append(np.nan)
        elif stock_data['SMA5'][i] < stock_data['SMA15'][i]:
            if position:
                signalBuy.append(np.nan)
                signalSell.append(stock_data['Adj Close'][i])
                position = False
            else:
                signalBuy.append(np.nan)
                signalSell.append(np.nan)
        else:
            signalBuy.append(np.nan)
            signalSell.append(np.nan)
    return pd.Series([signalBuy, signalSell])

# Buy and Sell signals for EMA
def buy_sellema(stock_data):
    signalBuyema = []
    signalSellema = []
    position = False

    for i in range(len(stock_data)):
        if stock_data['EMA5'][i] > stock_data['EMA15'][i]:
            if not position:
                signalBuyema.append(stock_data['Adj Close'][i])
                signalSellema.append(np.nan)
                position = True
            else:
                signalBuyema.append(np.nan)
                signalSellema.append(np.nan)
        elif stock_data['EMA5'][i] < stock_data['EMA15'][i]:
            if position:
                signalBuyema.append(np.nan)
                signalSellema.append(stock_data['Adj Close'][i])
                position = False
            else:
                signalBuyema.append(np.nan)
                signalSellema.append(np.nan)
        else:
            signalBuyema.append(np.nan)
            signalSellema.append(np.nan)
    return pd.Series([signalBuyema, signalSellema])

# Apply signals to stock data
stock_data['Buy_Signal_price'], stock_data['Sell_Signal_price'] = buy_sell(stock_data)
stock_data['Buy_Signal_priceEMA'], stock_data['Sell_Signal_priceEMA'] = buy_sellema(stock_data)

# To get latest close price
new = len(stock_data['Close'])-1
newupdate = stock_data['Close'][new]

# To get latest high price
newhigh = len(stock_data['High'])-1
newupdatehigh = stock_data['High'][newhigh]

# To get latest low price
newlow = len(stock_data['Low'])-1
newupdatelow = stock_data['Low'][newlow]

col1, col2, col3 = st.columns(3)
col1.metric("Close price on date", newupdate)
col2.metric("High price on date",  newupdatehigh)
col3.metric("Low price on date", newupdatelow)

# Display stock data in Streamlit
st.write(f"{stock_symbol} Stock Analysis")
st.write("## Stock Data")
st.write(stock_data)

# Plot stock data
st.write("## Stock Price and Signals")
st.line_chart(stock_data[['Adj Close', 'SMA5', 'SMA15', 'EMA5', 'EMA15']])
# Stock Volume
st.write("## Stock Volume")
st.line_chart(stock_data['Volume'])
#Bollinger Bands
st.write("## Bollinger Bands")
st.line_chart(pd.DataFrame({'Upper Bollinger Band': upper_band, 'Lower Bollinger Band': lower_band}))

#combined buy sell signals
st.write("## Buy/Sell Signals Combined")
fig2, ax = plt.subplots(figsize=(14, 8))
ax.plot(stock_data['Adj Close'] ,label = stock_symbol,linewidth=0.5, color='blue', alpha = 0.9)
ax.plot(stock_data['SMA5'], label = 'SMA5', linewidth=1,alpha = 0.85)
ax.plot(stock_data['SMA15'], label = 'SMA15' ,linewidth=1, alpha = 0.85)
ax.plot(stock_data['EMA5'], label = 'EMA5', linewidth=1,alpha = 0.85)
ax.plot(stock_data['EMA15'], label = 'EMA15', linewidth=1,alpha = 0.85)
ax.plot(upper_band, label='Upper Bollinger Band', color='red', linewidth=1.5)
ax.plot(lower_band, label='Lower Bollinger Band', color='green', linewidth=1.5)
ax.scatter(stock_data.index , stock_data['Buy_Signal_price'] , label = 'Buy SMA' , marker = '^', color = 'green',alpha =1 )
ax.scatter(stock_data.index , stock_data['Sell_Signal_price'] , label = 'Sell SMA' , marker = 'v', color = 'red',alpha =1 )
ax.scatter(stock_data.index , stock_data['Buy_Signal_priceEMA'] , label = 'Buy EMA' , marker = '^', color = 'black',alpha =1 )
ax.scatter(stock_data.index , stock_data['Sell_Signal_priceEMA'] , label = 'Sell' , marker = 'v', color = 'purple',alpha =1 )
ax.set_title(stock_symbol + " Price History with buy and sell signals",fontsize=10, backgroundcolor='blue', color='white')
ax.set_xlabel(f'{start_date} - {end_date}' ,fontsize=18)
ax.set_ylabel('Close Price INR (₨)' , fontsize=18)
legend = ax.legend()
ax.grid()
st.pyplot(fig2)


# Buy/Sell signals for SMA
st.write("## Buy/Sell Signals (SMA)")
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(stock_data['Adj Close'], label=stock_symbol, linewidth=0.5, color='blue', alpha=0.9)
ax.plot(stock_data['SMA5'], label='SMA5', linewidth=1, alpha=0.85)
ax.plot(stock_data['SMA15'], label='SMA15', linewidth=1, alpha=0.85)
ax.scatter(stock_data.index, stock_data['Buy_Signal_price'], label='Buy SMA', marker='^', color='green', alpha=1)
ax.scatter(stock_data.index, stock_data['Sell_Signal_price'], label='Sell SMA', marker='v', color='red', alpha=1)
ax.set_title(f"{stock_symbol} Price History with Buy and Sell Signals (SMA)", fontsize=10, backgroundcolor='blue',
             color='white')
ax.set_xlabel(f"{start_date} - {end_date}", fontsize=18)
ax.set_ylabel("Close Price INR (₨)", fontsize=18)
ax.grid()
st.pyplot(fig)

# Buy/Sell signals for EMA
st.write("## Buy/Sell Signals (EMA)")
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(stock_data['Adj Close'], label=stock_symbol, linewidth=0.5, color='blue', alpha=0.9)
ax.plot(stock_data['EMA5'], label='EMA5', linewidth=1, alpha=0.85)
ax.plot(stock_data['EMA15'], label='EMA15', linewidth=1, alpha=0.85)
ax.scatter(stock_data.index, stock_data['Buy_Signal_priceEMA'], label='Buy EMA', marker='^', color='black', alpha=1)
ax.scatter(stock_data.index, stock_data['Sell_Signal_priceEMA'], label='Sell EMA', marker='v', color='purple', alpha=1)
ax.set_title(f"{stock_symbol} Price History with Buy and Sell Signals (EMA)", fontsize=10, backgroundcolor='blue',
             color='white')
ax.set_xlabel(f"{start_date} - {end_date}", fontsize=18)
ax.set_ylabel("Close Price INR (₨)", fontsize=18)
ax.grid()
st.pyplot(fig)

# Buy/Sell signals for Bollinger
st.write("## Buy/Sell Signals with Bollinger Bands")
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(stock_data['Adj Close'], label=stock_symbol, linewidth=0.5, color='blue', alpha=0.9)
ax.plot(upper_band, label='Upper Bollinger Band', color='red', linewidth=1.5)
ax.plot(lower_band, label='Lower Bollinger Band', color='green', linewidth=1.5)
ax.set_title(f"{stock_symbol} Price History with bollinger bands", fontsize=10, backgroundcolor='blue',
             color='white')
ax.set_xlabel(f"{start_date} - {end_date}", fontsize=18)
ax.set_ylabel("Close Price INR (₨)", fontsize=18)
ax.grid()
st.pyplot(fig)

# Recommendations using TradingView API
st.write("## Recommendations")
symbol = stock_symbol.split('.')[0]  # Removing the exchange from the symbol
screener = "india"
exchange = "NSE"
interval = Interval.INTERVAL_1_MONTH

# Get recommendations for the stock
stock = TA_Handler(
    symbol=symbol,
    screener=screener,
    exchange=exchange,
    interval=interval,
)

recommendations = stock.get_analysis().summary

# Convert recommendations to a Pandas DataFrame
df = pd.DataFrame(recommendations, index=[0])

# Extract the relevant columns for the pie chart, handling missing columns
cols_to_plot = ['BUY', 'SELL', 'NEUTRAL', 'STRONG_BUY', 'STRONG_SELL']
existing_cols = [col for col in cols_to_plot if col in df.columns]
pie_data = df[existing_cols]

# Plot the pie chart
pie_data.T.plot.pie(subplots=True, autopct='%1.1f%%', legend=False, startangle=90)
plt.title(f"Recommendations for {symbol} on {exchange} - {interval}")
st.pyplot(plt)