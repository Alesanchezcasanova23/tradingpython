import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================
# RSI Calculation Function
# ======================================
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ======================================
# Streamlit App
# ======================================
st.title("Stock and RSI Indicator")

# Example list of possible tickers (you can expand this)
AVAILABLE_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ", "TLT", "VTI", "GLD", "XLF", "XLE"
]
# Select a ticker from predefined options
ticker = st.selectbox("Select a stock ticker for analysis:", options=AVAILABLE_TICKERS)


if ticker:
    try:
        stock_data = yf.download(ticker, period="1y")  # 1 year of data
        if stock_data.empty:
            st.error("Could not fetch data. Please check the ticker symbol.")
        else:
            stock_data['RSI'] = calculate_rsi(stock_data)

            # Create a subplot with 2 rows
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05,
                subplot_titles=(f"{ticker} RSI (Relative Strength Index)")
            )
        

            # Plot RSI
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=stock_data['RSI'], name="RSI", line_color="orange"),
                row=2, col=1
            )

            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

            fig.update_layout(height=700, showlegend=False, template="plotly_white")

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")