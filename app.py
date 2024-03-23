import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, timedelta
from tradingview_ta import TA_Handler, Interval, Exchange
import streamlit_shadcn_ui as ui
from local_components import card_container
import plotly.express as px
import plotly.graph_objects as go
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
import openai
from IPython.display import Markdown, display
import time 




st.set_page_config(
    page_title="Veracity",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(':blue[Veracity]wise')
#st.link_button("GitHub", "https://github.com/aryaneelshivam/ArysStockAnalysis")
st.write(
    """
    ![Static Badge](https://img.shields.io/badge/%20version-Beta-white)
    """
)

#setting up OpenAI Api key
llm = OpenAI(api_token=st.secrets["OpenAI_Key"])
openai.api_key = st.secrets["OpenAI_Key"]
#tab = ui.tabs(options=['Local file', 'Google sheets', 'Airtable', 'Snowflake'], default_value='Local file', key="select")

# Set Matplotlib style
#plt.style.use('fivethirtyeight')
yf.pdr_override()

# Sidebar for user input
st.sidebar.markdown("""
<a href="https://www.producthunt.com/posts/arys-analysis?utm_source=badge-featured&utm_medium=badge&utm_souce=badge-arys&#0045;analysis" target="_blank">
    <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=432960&theme=light"
         alt="Arys&#0032;analysis - Open&#0045;sourced&#0044;&#0032;stock&#0032;market&#0032;analysis&#0032;and&#0032;visualization&#0032;webapp | Product Hunt"
         style="width: 250px; height: 54px;" width="250" height="54" />
</a>
""",  unsafe_allow_html=True)

st.sidebar.caption('We are also available on producthunt! Upvoting the software really helps!')
stock_symbol = st.sidebar.text_input("Enter Stock Symbol")
st.sidebar.caption('Enter stock symbol from https://finance.yahoo.com/')
# Date Range Selection
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=60))
end_date = st.sidebar.date_input("End Date", date.today())
st.sidebar.caption("⚠ NOTE: Make sure to keep a minimum of 30-day gap between the start-date and the end-date.")
usekey = st.sidebar.text_input("Enter private use-key", placeholder="Enter your private use-key", key="placeholder", type="password")
buybutton = st.sidebar.link_button("Get your Key", "https://teenscript.substack.com/", type="primary", help="Purchase your private use-key to work with droptable.", use_container_width=True)
st.sidebar.caption('If you dont have a private use-key, then get one and keep it safe.')
st.sidebar.link_button("Read the guide docs 📄", "https://docs.google.com/document/d/1DezoHwpJB_qJ9kalaaLAhi1zHLG_KwcUq65Biiiuzqw/edit?usp=sharing", use_container_width=True)
sensitivity = 0.03
if usekey == "admin1818":
    useaccess = usekey
else:
    st.warning("Enter correct use-key")

if stock_symbol:
    try:
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


        #Support-Resistance logic
        support_levels = []
        resistance_levels = []

        for i in range(1, len(stock_data['Close']) - 1):
            previous_close = stock_data['Close'][i - 1]
            current_close = stock_data['Close'][i]
            next_close = stock_data['Close'][i + 1]

            if current_close < previous_close and current_close < next_close:
                support_levels.append(current_close)
            elif current_close > previous_close and current_close > next_close:
                resistance_levels.append(current_close)

        # Filter levels based on sensitivity
        support_levels = [level for level in support_levels if any(abs(level - s) > sensitivity * level for s in support_levels)]
        resistance_levels = [level for level in resistance_levels if any(abs(level - r) > sensitivity * level for r in resistance_levels)]

        # Apply signals to stock data
        stock_data['Buy_Signal_price'], stock_data['Sell_Signal_price'] = buy_sell(stock_data)
        stock_data['Buy_Signal_priceEMA'], stock_data['Sell_Signal_priceEMA'] = buy_sellema(stock_data)

        # To get latest close price
        new = len(stock_data['Close'])-1
        newupdate = round(stock_data['Close'][new],2)

        # To get latest high price
        newhigh = len(stock_data['High'])-1
        newupdatehigh = round(stock_data['High'][newhigh],2)

        # To get latest low price
        newlow = len(stock_data['Low'])-1
        newupdatelow = round(stock_data['Low'][newlow],2)

        cols = st.columns(3)
        with cols[0]:
            ui.metric_card(title="Close price on date", content=newupdate, description="The retrieved closing price ☝", key="card1")
        with cols[1]:
            ui.metric_card(title="High price on date", content=newupdatehigh, description="The retrieved high price ☝", key="card2")
        with cols[2]:
            ui.metric_card(title="Low price on date", content=newupdatelow, description="The retrieved low price ☝", key="card3")


        # Display stock data in Streamlit
        with card_container():
            st.write(f"Stock in view ➡ {stock_symbol}")
        with st.expander("Expand to view retrieved stock table"):
            st.dataframe(stock_data)

        #columns for AI boxes
        box1,box2 = st.columns(2)
        with box1:
            user_input = st.text_area("Enter your input 💬", placeholder="Enter your question/query", height=200)  
            enter_button = st.button("Enter 💣", use_container_width=True, type="primary", disabled=not useaccess)
            querydata = PandasQueryEngine(df=stock_data, verbose=True, synthesize_response=True)
            if enter_button:
                if user_input:
                    with st.spinner():
                        conv = querydata.query(user_input)

        with box2:
            output = st.text_area("Your generated output 🎉", placeholder="The output will be displayed here", value=conv if 'conv' in locals() else "", height=200)
            generate = st.button("Generate AI report ⚡", use_container_width=True, disabled=not useaccess)

        #full AI technical analysis logic
        if generate:
            query_engine = PandasQueryEngine(df=stock_data, verbose=True, synthesize_response=True)
            with st.spinner("Exploring data..."):
                response = query_engine.query("take all the datapoints and recognize possible patterns.")
            if response:
                with st.spinner("Analysing data..."):
                    response2 = query_engine.query("take all the datapoints and generate a market analysis.")
            if response2:
                with st.spinner("Generating summary..."):
                    response1 = query_engine.query("take all the datapoints and generate a market forecast.")
            if response1:
                with card_container():
                    st.info(response2, icon="💡")
                with card_container():
                    st.info(response1, icon="🎯")
                with card_container():
                    st.info(response, icon="📌")


        with card_container():
            # Stock Volume
            color = "blue"
            st.write("### Stock Volume")
            fig = px.bar(stock_data['Volume'], color=stock_data['Volume'])
            fig.update_traces(marker_line_width=1)
            st.plotly_chart(fig)



        with card_container():

            fig2 = go.Figure()

            # Plotting stock data
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name=stock_symbol, line=dict(color='blue', width=0.5)))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA5'], mode='lines', name='SMA5', line=dict(color='blue', width=1)))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA15'], mode='lines', name='SMA15', line=dict(color='blue', width=1)))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA5'], mode='lines', name='EMA5', line=dict(color='blue', width=1)))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA15'], mode='lines', name='EMA15', line=dict(color='blue', width=1)))

            # Plotting Bollinger Bands
            fig2.add_trace(go.Scatter(x=stock_data.index, y=upper_band, mode='lines', name='Upper Bollinger Band', line=dict(color='red', width=1.5)))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=lower_band, mode='lines', name='Lower Bollinger Band', line=dict(color='green', width=1.5)))

            # Plotting support and resistance levels
            if support_levels:
                last_support_level = support_levels[-1]
                fig2.add_shape(type="line", x0=stock_data.index[0], y0=last_support_level, x1=stock_data.index[-1], y1=last_support_level, line=dict(color="green", width=0.8), name=f'Last Support Level: {last_support_level}')
            if resistance_levels:
                last_resistance_level = resistance_levels[-1]
                fig2.add_shape(type="line", x0=stock_data.index[0], y0=last_resistance_level, x1=stock_data.index[-1], y1=last_resistance_level, line=dict(color="red", width=0.8), name=f'Last Resistance Level: {last_resistance_level}')

            # Plotting buy and sell signals
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Buy_Signal_price'], mode='markers', name='Buy SMA', marker=dict(symbol='triangle-up', color='green')))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Sell_Signal_price'], mode='markers', name='Sell SMA', marker=dict(symbol='triangle-down', color='red')))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Buy_Signal_priceEMA'], mode='markers', name='Buy EMA', marker=dict(symbol='triangle-up', color='black')))
            fig2.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Sell_Signal_priceEMA'], mode='markers', name='Sell EMA', marker=dict(symbol='triangle-down', color='purple')))

            # Update layout
            fig2.update_layout(title=stock_symbol + " Price History with buy and sell signals",
                               xaxis_title=f'{start_date} - {end_date}',
                               yaxis_title='Close Price INR (₨)',
                               legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="left", x=1),
                               showlegend=True,
                               plot_bgcolor='white'
                               )

            st.plotly_chart(fig2)



        # Buy/Sell signals for SMA

        fig = go.Figure()
        with card_container():
            # Plotting stock data
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name=stock_symbol, line=dict(color='blue', width=0.5)))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA5'], mode='lines', name='SMA5', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA15'], mode='lines', name='SMA15', line=dict(color='blue', width=1)))

            # Plotting support and resistance levels
            if support_levels:
                last_support_level = support_levels[-1]
                fig.add_shape(type="line", x0=stock_data.index[0], y0=last_support_level, x1=stock_data.index[-1], y1=last_support_level, line=dict(color="green", width=0.8), name=f'Last Support Level: {last_support_level}')
            if resistance_levels:
                last_resistance_level = resistance_levels[-1]
                fig.add_shape(type="line", x0=stock_data.index[0], y0=last_resistance_level, x1=stock_data.index[-1], y1=last_resistance_level, line=dict(color="red", width=0.8), name=f'Last Resistance Level: {last_resistance_level}')

            # Plotting buy and sell signals
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Buy_Signal_price'], mode='markers', name='Buy SMA', marker=dict(symbol='triangle-up', color='green')))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Sell_Signal_price'], mode='markers', name='Sell SMA', marker=dict(symbol='triangle-down', color='red')))

            # Update layout
            fig.update_layout(title=f"{stock_symbol} Price History with Buy and Sell Signals (SMA)",
                              xaxis_title=f"{start_date} - {end_date}",
                              yaxis_title="Close Price INR (₨)",
                              legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="left", x=1),
                              showlegend=True,
                              plot_bgcolor='white'
                              )

            st.plotly_chart(fig)


        expander = st.expander("See explanation of above indicators")
        expander.write('''
            The basic idea of **SMA crossover strategy** is to look for the intersections of two SMAs with different periods: 
            a *fast SMA* and a *slow SMA.* The fast SMA is more responsive to the price movements, while the slow SMA is more stable and smooth. 
            *When the fast SMA crosses above the slow SMA, it is a bullish signal, **indicating that the price is likely to go up.***
            *When the fast SMA crosses below the slow SMA, **it is a bearish signal, indicating that the price is likely to go down.***
        ''')

        # Buy/Sell signals for EMA
        with card_container():
            fig = go.Figure()

            # Plotting stock data
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name=stock_symbol, line=dict(color='blue', width=0.5)))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA5'], mode='lines', name='EMA5', line=dict(color='blue', width=1)))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA15'], mode='lines', name='EMA15', line=dict(color='blue', width=1)))

            # Plotting support and resistance levels
            if support_levels:
                last_support_level = support_levels[-1]
                fig.add_shape(type="line", x0=stock_data.index[0], y0=last_support_level, x1=stock_data.index[-1], y1=last_support_level, line=dict(color="green", width=0.8), name=f'Last Support Level: {last_support_level}')
            if resistance_levels:
                last_resistance_level = resistance_levels[-1]
                fig.add_shape(type="line", x0=stock_data.index[0], y0=last_resistance_level, x1=stock_data.index[-1], y1=last_resistance_level, line=dict(color="red", width=0.8), name=f'Last Resistance Level: {last_resistance_level}')

            # Plotting buy and sell signals
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Buy_Signal_priceEMA'], mode='markers', name='Buy EMA', marker=dict(symbol='triangle-up', color='black')))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Sell_Signal_priceEMA'], mode='markers', name='Sell EMA', marker=dict(symbol='triangle-down', color='purple')))

            # Update layout
            fig.update_layout(title=f"{stock_symbol} Price History with Buy and Sell Signals (EMA)",
                              xaxis_title=f"{start_date} - {end_date}",
                              yaxis_title="Close Price INR (₨)",
                              legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="left", x=1),
                              showlegend=True,
                              plot_bgcolor='white'
                              )

            st.plotly_chart(fig)

        expander = st.expander("See explanation of above indicators")
        expander.write('''
            The basic idea of **EMA crossover strategy** is to look for the intersections of two EMAs with different periods: 
            a *fast EMA* and a *slow EMA.* The fast EMA is more responsive to the price movements, while the slow EMA is more stable and smooth. 
            *When the fast EMA crosses above the slow EMA, it is a bullish signal, **indicating that the price is likely to go up.***
            *When the fast EMA crosses below the slow EMA, **it is a bearish signal, indicating that the price is likely to go down.***
        ''')

        # Buy/Sell signals for Bollinger
        with card_container():
            fig = go.Figure()

            # Plotting stock data
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name=stock_symbol, line=dict(color='blue', width=0.5)))
            fig.add_trace(go.Scatter(x=stock_data.index, y=upper_band, mode='lines', name='Upper Bollinger Band', line=dict(color='red', width=1.5)))
            fig.add_trace(go.Scatter(x=stock_data.index, y=lower_band, mode='lines', name='Lower Bollinger Band', line=dict(color='green', width=1.5)))

            # Plotting support and resistance levels
            if support_levels:
                last_support_level = support_levels[-1]
                fig.add_shape(type="line", x0=stock_data.index[0], y0=last_support_level, x1=stock_data.index[-1], y1=last_support_level, line=dict(color="green", width=0.8), name=f'Last Support Level: {last_support_level}')
            if resistance_levels:
                last_resistance_level = resistance_levels[-1]
                fig.add_shape(type="line", x0=stock_data.index[0], y0=last_resistance_level, x1=stock_data.index[-1], y1=last_resistance_level, line=dict(color="red", width=0.8), name=f'Last Resistance Level: {last_resistance_level}')

            # Update layout
            fig.update_layout(title=f"{stock_symbol} Price History with Bollinger Bands",
                              xaxis_title=f"{start_date} - {end_date}",
                              yaxis_title="Close Price INR (₨)",
                              legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="left", x=1),
                              showlegend=True,
                              plot_bgcolor='white'
                              )

            st.plotly_chart(fig)

        expander = st.expander("See explanation of above indicators")
        expander.write('''
            A common Bollinger Bands® strategy is to look for **overbought and oversold conditions in the market.** 
            *When the price touches or exceeds the upper band, it may indicate that the **security is overbought** and due for a pullback.* 
            Conversely, *when the price touches or falls below the lower band, it may indicate that the **security is oversold** and ready for a bounce.*
        ''')

        #Buy/Sell support-resistance
        with card_container():
            fig = go.Figure()

            # Plotting stock data
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name=stock_symbol, line=dict(color='blue', width=0.5)))

            # Plotting support and resistance levels
            if support_levels:
                last_support_level = support_levels[-1]
                fig.add_shape(type="line", x0=stock_data.index[0], y0=last_support_level, x1=stock_data.index[-1], y1=last_support_level, line=dict(color="green", width=0.8), name=f'Last Support Level: {last_support_level}')
            if resistance_levels:
                last_resistance_level = resistance_levels[-1]
                fig.add_shape(type="line", x0=stock_data.index[0], y0=last_resistance_level, x1=stock_data.index[-1], y1=last_resistance_level, line=dict(color="red", width=0.8), name=f'Last Resistance Level: {last_resistance_level}')

            # Update layout
            fig.update_layout(title=f"{stock_symbol} Price History with Support-Resistance levels",
                              xaxis_title=f"{start_date} - {end_date}",
                              yaxis_title="Close Price INR (₨)",
                              legend=dict(orientation="v", yanchor="top", y=1.02, xanchor="left", x=1),
                              showlegend=True,
                              plot_bgcolor='white'
                              )

            st.plotly_chart(fig)

        #Support-Resistance explainer
        expander = st.expander("See explanation of above indicators")
        expander.write('''
            **Price support occurs when a surplus of buying activity occurs when an asset’s price drops to a particular area.** 
            This buying activity causes the *price to move back up and away from the support level.* Resistance is the opposite of support. 
            Resistance levels are areas where **prices fall due to overwhelming selling pressure.**
        ''')

        # Recommendations using TradingView API
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

        with card_container():
            # Plot the pie chart
            fig = go.Figure(data=[go.Pie(labels=pie_data.columns, values=pie_data.iloc[0], textinfo='label+percent')])
            fig.update_traces(hole=0.4, hoverinfo="label+percent", textinfo="value", marker=dict(colors=['green', 'red', 'orange', 'blue', 'purple'], line=dict(color='#000000', width=0)))
            fig.update_layout(title=f"Recommendations for {symbol} on {exchange} - {interval}")

            st.plotly_chart(fig)
    except:
        st.warning("Wrong Stock symbol, check yahoo finance website for symbols.", icon="⚠")
else:
    st.error("Enter a valid stock symbol from Yahoo finance to continue", icon="🚨")
