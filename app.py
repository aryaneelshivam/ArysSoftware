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
import time 
import streamlit.components.v1 as components
#import pandas_ta as ta


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
#llm = OpenAI(api_token=st.secrets["OpenAI_Key"])
#openai.api_key = st.secrets["OpenAI_Key"]
#tab = ui.tabs(options=['Local file', 'Google sheets', 'Airtable', 'Snowflake'], default_value='Local file', key="select")

# Set Matplotlib style
#plt.style.use('fivethirtyeight')
yf.pdr_override()

# Sidebar for user input
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", placeholder="Ex: TATASTEEL.NS")
# Date Range Selection
start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=60))
end_date = st.sidebar.date_input("End Date", date.today())
st.sidebar.caption("⚠ NOTE: Make sure to keep a minimum of 30-day gap between the start-date and the end-date.")
#usekey = st.sidebar.text_input("Enter private use-key", placeholder="Enter your private use-key", key="placeholder", type="password")
#buybutton = st.sidebar.link_button("Get your Key", "https://teenscript.substack.com/", type="primary", help="Purchase your private use-key to work with veracity.", use_container_width=True, disabled=True)
#st.sidebar.caption('If you dont have a private use-key, then get one and keep it safe.')
st.sidebar.link_button("Read the guide docs 📄", "https://docs.google.com/document/d/1DezoHwpJB_qJ9kalaaLAhi1zHLG_KwcUq65Biiiuzqw/edit?usp=sharing", use_container_width=True)
sensitivity = 0.03
with st.popover("Open Google Trends popover 📈"):
            st.markdown("##### Google trends: rising search terms over the last 7 days")
            components.html("""<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/3620_RC01/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("RELATED_QUERIES", {"comparisonItem":[{"keyword":"share price","geo":"IN","time":"now 7-d"}],"category":7,"property":""}, {"exploreQuery":"cat=7&date=now%207-d&geo=IN&q=share%20price&hl=en-GB","guestPath":"https://trends.google.co.in:443/trends/embed/"}); </script>""", height=400)


if stock_symbol:
        # Download stock data
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        # additional stock_info for AI analysis
        stock_details = yf.Ticker(stock_symbol)
        # Calculate technical indicators
        rstd = stock_data['Close'].rolling(window=15).std()
        stock_data['EMA5'] = stock_data['Close'].ewm(span=5).mean()
        stock_data['EMA15'] = stock_data['Close'].ewm(span=15).mean()
        stock_data['SMA5'] = stock_data['Close'].rolling(window=5).mean()
        stock_data['SMA15'] = stock_data['Close'].rolling(window=15).mean()
        upper_band = stock_data['SMA15'] + 2 * rstd
        lower_band = stock_data['SMA15'] - 2 * rstd

        # Calculate technical indicators using pandas-ta
        #stock_data.ta.macd(append=True)
        #stock_data.ta.rsi(append=True)
        #stock_data.ta.bbands(append=True)
        #stock_data.ta.obv(append=True)

        # Calculate Stochastic Oscillator
        high_14 = stock_data['High'].rolling(window=5).max()
        low_14 = stock_data['Low'].rolling(window=5).min()
        stock_data['%K'] = 100 * ((stock_data['Close'] - low_14) / (high_14 - low_14))
        stock_data['%D'] = stock_data['%K'].rolling(window=3).mean()


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

        #stochastic signals logic
        def calculate_stochastic_signals(stock_data):
            stochBuy = []
            stochSell = []
            position = False

            for i in range(len(stock_data)):
                if stock_data['%K'][i] < 20 and stock_data['%D'][i] < 20 and stock_data['%K'][i] > stock_data['%D'][i]:
                    if not position:
                        stochBuy.append(stock_data['Adj Close'][i])
                        stochSell.append(np.nan)
                        position = True
                    else:
                        stochBuy.append(np.nan)
                        stochSell.append(np.nan)
                elif stock_data['%K'][i] > 80 and stock_data['%D'][i] > 80 and stock_data['%K'][i] < stock_data['%D'][i]:
                    if position:
                        stochBuy.append(np.nan)
                        stochSell.append(stock_data['Adj Close'][i])
                        position = False
                    else:
                        stochBuy.append(np.nan)
                        stochSell.append(np.nan)
                else:
                    stochBuy.append(np.nan)
                    stochSell.append(np.nan)
            return pd.Series([stochBuy, stochSell])

        #RSI logic
        def calculate_rsi(data, window=5):
            delta = stock_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

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

        # Add buy and sell signals to the DataFrame
        stock_data['Stoch_Buy'], stock_data['Stoch_Sell'] = calculate_stochastic_signals(stock_data)
        stock_data['RSI'] = calculate_rsi(stock_data)

        cols = st.columns(3)
        with cols[0]:
            ui.metric_card(title="Close price on date", content=newupdate, description="The retrieved closing price ☝", key="card1")
        with cols[1]:
            ui.metric_card(title="High price on date", content=newupdatehigh, description="The retrieved high price ☝", key="card2")
        with cols[2]:
            ui.metric_card(title="Low price on date", content=newupdatelow, description="The retrieved low price ☝", key="card3")


        # Display stock data in Streamlit
        datastore = stock_details.info
        st.info(f"Stock in view ➡ {stock_symbol}",icon="📢")
        st.info(stock_details.info["longBusinessSummary"], icon="💡")
        try:
            profitmargins = stock_details.info["profitMargins"]
            net_income_to_common = stock_details.info["netIncomeToCommon"]
            enterprise_to_ebitda = stock_details.info["enterpriseToEbitda"]
            TotalCash = stock_details.info["totalCash"]
            TotalDebt = stock_details.info["totalDebt"]
            Ebidta_margins = stock_details.info["ebitdaMargins"]
            Operations_margins = stock_details.info["operatingMargins"]
            DebtToEquity = stock_details.info["debtToEquity"]
            ai_df = pd.DataFrame.from_dict(stock_details.info)
        except:
            st.error("Error loading additional information",icon="🚨")
            


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

        ss1, ss2 = st.columns(2)
        # Plot Stochastic Oscillator
        fig3 = go.Figure()
        with ss1:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%K'], mode='lines', name='%K', line=dict(color='blue')))
            fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['%D'], mode='lines', name='%D', line=dict(color='red')))
            fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Stoch_Buy'], name='Buy Signal (Stochastic)', mode='markers', marker=dict(color='green', symbol='triangle-up')))
            fig3.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Stoch_Sell'], name='Sell Signal (Stochastic)', mode='markers', marker=dict(color='red', symbol='triangle-down')))
            fig3.update_layout(title='Stochastic Oscillator',
                               xaxis_title='Date',
                               yaxis_title='Value',
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig3)

        with ss2:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
            fig_rsi.update_layout(title='Relative Strength Index (RSI)',
                                  xaxis_title='Date',
                                  yaxis_title='RSI Value',
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

            # Display plots
            st.plotly_chart(fig_rsi)

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
            try:
                # Plot the pie chart
                fig = go.Figure(data=[go.Pie(labels=pie_data.columns, values=pie_data.iloc[0], textinfo='label+percent')])
                fig.update_traces(hole=0.4, hoverinfo="label+percent", textinfo="value", marker=dict(colors=['green', 'red', 'orange', 'blue', 'purple'], line=dict(color='#000000', width=0)))
                fig.update_layout(title=f"Recommendations for {symbol} on {exchange} - {interval}")

                st.plotly_chart(fig)
            except:
                st.warning("Cannot load stock recommendation for"+stock_symbol)

