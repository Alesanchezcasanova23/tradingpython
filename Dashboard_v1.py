import streamlit as st

# Data Handling and Processing
import pandas as pd
import numpy as np

# Visualization
import plotly.graph_objects as go
import matplotlib.dates as mdates
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Financial Data
import yfinance as yf

# Time Series Analysis
from arch import arch_model

# Statistical Tests and Metrics
from scipy.stats import norm  # for probability calculations
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")  # Ignore convergence warnings

# Lightweight Charts (Web-Based Visualization)
from plotly.subplots import make_subplots


#=======================================================================================================================
# Start of the code
#=======================================================================================================================

def combine_data(*dataframes_with_symbols):
    """
    Combine multiple stock datasets based on their index. Each dataset must have a 'Date' column
    or a DatetimeIndex. The stock symbols are passed explicitly.
    """
    combined_data = None

    for df, stock_symbol in dataframes_with_symbols:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Each input should be a pandas DataFrame.")

        # Ensure the DataFrame has a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
            else:
                raise ValueError("DataFrame must have a 'Date' column or a DatetimeIndex.")

        # Optionally drop 'Date' column if it exists as a regular column
        if 'Date' in df.columns:
            df = df.drop(columns=['Date'])

        # Rename columns to include the stock symbol
        df.columns = [f"{col}_{stock_symbol}" for col in df.columns]

        # Concatenate along the index using an inner join to keep only common dates
        if combined_data is None:
            combined_data = df
        else:
            combined_data = pd.concat([combined_data, df], axis=1, join='inner')

    return combined_data

# =======================================================================================================================
# Streamlit title
# =======================================================================================================================

st.title("Stock Analysis Dashboard v1")
st.subheader('----------------------------------------------------------')

# =======================================================================================================================
# Main Stock Input boton
# =======================================================================================================================

# Initialize a list in session state to store tickers if it doesn't exist
if "stock_list" not in st.session_state:
    st.session_state.stock_list = []

# Example list of possible tickers (you can expand this)
AVAILABLE_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ", "TLT", "VTI", "GLD", "XLF", "XLE"
]
# Select a ticker from predefined options
ticker = st.selectbox("Select a stock ticker for analysis:", options=AVAILABLE_TICKERS)

# Button to add ticker to the list
if st.button("Add Stock"):
    if ticker:
        if ticker.upper() not in st.session_state.stock_list:
            st.session_state.stock_list.append(ticker.upper())
            st.success(f"Added {ticker.upper()} to the analysis list!")
        else:
            st.warning(f"{ticker.upper()} is already in the list.")
# Display tickers along with a delete button for each
if st.session_state.stock_list:
    st.write("Tickers for analysis:")
    for i, t in enumerate(st.session_state.stock_list):
        col1, col2 = st.columns([3, 1])
        col1.write(t)
        if col2.button("Delete", key=f"delete_{i}"):
            st.session_state.stock_list.pop(i)
            try:
                st.experimental_rerun()
            except st.runtime.scriptrunner.script_run_context.RerunException:
                pass

# Create a list to store processed data for each ticker
processed_data_list = []
information = []

# Process each ticker for analysis
if st.session_state.stock_list:
    for t in st.session_state.stock_list:
        try:
            # Download all available historical data
            stock_data = yf.download(t, period="max")
            if stock_data.empty:
                st.error(f"Error: {t} has been delisted or was not found. Please try a different ticker.")
                continue
            else:
                st.success(f"Stock data for {t} fetched successfully!")

                # Clean the data
                stock_data.columns = [col[0] for col in stock_data.columns]  # Flatten multi-level columns if any
                stock_data.reset_index(inplace=True)
                stock_data['Date'] = pd.to_datetime(stock_data['Date'].apply(lambda x: x.date()))
                stock_data['Volume'] = stock_data['Volume'].astype(float)


                stock_data['Return'] = stock_data['Close'].pct_change()
                stock_data.dropna(subset=['Return'], inplace=True)
                stock_data['Cumulative Return'] = (1 + stock_data['Return']).cumprod() - 1

                numerical_stats = stock_data.drop(columns=['Date','Cumulative Return']).agg(
                    ['mean', 'median', 'count', 'std', 'max', 'min', 'var','kurt','skew']
                ).round(2)

                st.header(f"Statistics for {t}")
                st.write(numerical_stats)

#=======================================================================================================================
                #plotting candletick chart
#=======================================================================================================================

                st.title(f"Stock Chart for {t}")

                # Set title using ticker_name if provided, else fallback to DataFrame attribute
                title = t if t else stock_data.attrs.get("name", "Stock")

                # Determine the current price as the last closing price in the dataset
                current_price = stock_data['Close'].iloc[-1]

                # Create subplots: 2 rows, one for candlestick, one for volume
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03,
                                    row_heights=[0.7, 0.3],
                                    subplot_titles=(f'Candlestick Chart of {title}', 'Volume'))

                # Add candlestick trace
                fig.add_trace(
                    go.Candlestick(
                        x=stock_data['Date'],
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        increasing_line_color='green',
                        decreasing_line_color='red',
                        name='Price'
                    ),
                    row=1, col=1
                )

                # Add volume as a bar chart
                fig.add_trace(
                    go.Bar(
                        x=stock_data['Date'],
                        y=stock_data['Volume'],
                        marker_color='blue',
                        name='Volume'
                    ),
                    row=2, col=1
                )

                # Add a horizontal line for the current price on the candlestick chart
                fig.add_shape(
                    type="line",
                    x0=stock_data['Date'].min(),
                    y0=current_price,
                    x1=stock_data['Date'].max(),
                    y1=current_price,
                    line=dict(color="RoyalBlue", width=2, dash="dot"),
                    row=1, col=1
                )

                # Annotate the current price on the chart
                fig.add_annotation(
                    x=stock_data['Date'].iloc[-1],
                    y=current_price,
                    text=f"Current Price: {current_price:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30,
                    row=1, col=1
                )

                # Update layout
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Price',
                    xaxis_rangeslider_visible=False,  # Hide range slider
                    template='plotly_white'
                )

                # Plot the chart in Streamlit
                st.plotly_chart(fig)



#=======================================================================================================================
#               Buy/Sell Recommendation (EMA Strat)
#=======================================================================================================================

                st.title("Buy/Sell Reccomendation (not financial advice)")

                conf_int = 0.95

                # Before running the EMA strategy, ensure the DataFrame has a "Date" column
                if 'Date' not in stock_data.columns:
                    stock_data = stock_data.reset_index()  # Bring the Date index back as a column

                if isinstance(stock_data.columns, pd.MultiIndex):
                    stock_data.columns = stock_data.columns.get_level_values(0)

                stock_data['Date'] = pd.to_datetime(stock_data['Date'])
                stock_data.set_index('Date', inplace=True)
                stock_data.sort_index(inplace=True)

                # Drop rows with NaN values resulting from the shift
                stock_data.dropna(subset=['Close'], inplace=True)

                # Calculate the EMA and EWM Std Dev
                ema = stock_data['Close'].ewm(span=30, adjust=False).mean()
                ewm_std = stock_data['Close'].ewm(span=30, adjust=False).std()

                # Calculate Z-scores
                z_scores = (stock_data['Close'] - ema) / ewm_std

                # Calculate probability of an upward move
                stock_data['Probability Up'] = 1 - norm.cdf(z_scores)

                # Define dynamic thresholds based on probabilities
                z_threshold = norm.ppf(1 - conf_int)
                upper_boundary = ema + (z_threshold * ewm_std)
                lower_boundary = ema - (z_threshold * ewm_std)


                # Calculate the Volume-weighted RSI
                def calculate_rsi(stock_data, period=14):
                    delta = stock_data['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    volume_weighted_gain = (gain * stock_data['Volume']).rolling(window=period).mean()
                    volume_weighted_loss = (loss * stock_data['Volume']).rolling(window=period).mean()
                    rs = volume_weighted_gain / volume_weighted_loss
                    rsi = 100 - (100 / (1 + rs))
                    return rsi


                stock_data['RSI'] = calculate_rsi(stock_data, period=14)

                # Define trend based on short-term and long-term EMAs
                stock_data['short_term_ema'] = stock_data['Close'].ewm(span=30, adjust=False).mean()
                stock_data['long_term_ema'] = stock_data['Close'].ewm(span=200, adjust=False).mean()
                stock_data['trend'] = np.where(stock_data['short_term_ema'] > stock_data['long_term_ema'], 'uptrend',
                                               'downtrend')

                # Apply weighted Z-score based on trend direction
                stock_data['z_score_weighted'] = np.where(stock_data['trend'] == 'downtrend', z_scores * 1.5,
                                                          z_scores * 0.5)


                # Calculate MACD with Volume weighting
                def calculate_macd(stock_data, fast_period=12, slow_period=26, signal_period=9):
                    fast_ema = stock_data['Close'].ewm(span=fast_period, adjust=False).mean()
                    slow_ema = stock_data['Close'].ewm(span=slow_period, adjust=False).mean()
                    macd = fast_ema - slow_ema
                    signal_line = macd.ewm(span=signal_period, adjust=False).mean()
                    weighted_signal = signal_line * (1 + stock_data['Volume'] / stock_data['Volume'].mean())
                    return macd, weighted_signal


                stock_data['macd'], stock_data['macd_signal'] = calculate_macd(stock_data)


                # Calculate GARCH-based volatility prediction (with scaling)
                def calculate_garch_volatility(stock_data, scaling_factor=100):
                    returns_series = stock_data['Close'].pct_change().dropna()
                    returns_rescaled = returns_series * scaling_factor
                    garch_model = arch_model(returns_rescaled, vol='Garch', p=1, q=1)
                    garch_fit = garch_model.fit(disp="off")
                    predicted_volatility_scaled = garch_fit.conditional_volatility
                    predicted_volatility = predicted_volatility_scaled / scaling_factor
                    return predicted_volatility


                stock_data['predicted_volatility'] = calculate_garch_volatility(stock_data, scaling_factor=100)


                # Calculate On-Balance Volume (OBV)
                def calculate_obv(stock_data):
                    obv = [0] * len(stock_data)
                    for i in range(1, len(stock_data)):
                        if stock_data['Close'].iloc[i] > stock_data['Close'].iloc[i - 1]:
                            obv[i] = obv[i - 1] + stock_data['Volume'].iloc[i]
                        elif stock_data['Close'].iloc[i] < stock_data['Close'].iloc[i - 1]:
                            obv[i] = obv[i - 1] - stock_data['Volume'].iloc[i]
                        else:
                            obv[i] = obv[i - 1]
                    stock_data['OBV'] = obv
                    return stock_data


                stock_data = calculate_obv(stock_data)
                # Initialize signal state
                last_signal = "NONE"  # initial state: no position

                # Create empty lists to store signal dates and prices
                buy_signal_dates = []
                buy_signal_prices = []

                sell_signal_dates = []
                sell_signal_prices = []

                # Iterate over the stock data in order
                for row in stock_data.itertuples():
                    date = row.Index
                    close = row.Close
                    rsi = row.RSI
                    macd = row.macd
                    macd_signal = row.macd_signal
                    volatility = row.predicted_volatility

                    # Current dynamic thresholds
                    lower = lower_boundary.loc[date]
                    upper = upper_boundary.loc[date]

                    # ----- BUY condition -----
                    if last_signal in ["SELL", "NONE"]:
                        if (close < lower) and (rsi < 30) and (macd > macd_signal) and (volatility < stock_data['predicted_volatility'].quantile(0.75)):
                            # Emit BUY
                            buy_signal_dates.append(date)
                            buy_signal_prices.append(row.Close)  # use Adjusted column for plotting
                            last_signal = "BUY"
                            continue  # after buy, skip to next date

                    # ----- SELL condition -----
                    if last_signal == "BUY":
                        if (close > upper) and (rsi > 70) and (macd < macd_signal * 1.5) and (volatility < stock_data['predicted_volatility'].quantile(0.75)):
                            # Emit SELL
                            sell_signal_dates.append(date)
                            sell_signal_prices.append(row.Close)  # use Adjusted column for plotting
                            last_signal = "SELL"
                            continue  # after sell, skip to next date

                # Create the Plotly figure for the EMA strategy
                fig4 = go.Figure()

                # Add Close Price Line
                fig4.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
                                          mode='lines', name='Close Price',
                                          line=dict(color='blue', width=2), opacity=0.6))

                # Add 30-Day EMA
                fig4.add_trace(go.Scatter(x=stock_data.index, y=ema,
                                          mode='lines', name='30-Day EMA',
                                          line=dict(color='red', dash='dash'), opacity=0.8))

                # Add Upper and Lower Boundaries
                fig4.add_trace(go.Scatter(x=stock_data.index, y=upper_boundary,
                                          mode='lines', name=f'+{int(conf_int * 100)}% Threshold',
                                          line=dict(color='green', dash='dash'), opacity=0.8))

                fig4.add_trace(go.Scatter(x=stock_data.index, y=lower_boundary,
                                          mode='lines', name=f'-{int(conf_int * 100)}% Threshold',
                                          line=dict(color='orange', dash='dash'), opacity=0.8))

                # Add MACD Histogram
                fig4.add_trace(go.Bar(x=stock_data.index, y=stock_data['macd'] - stock_data['macd_signal'],
                                      name='MACD Histogram', marker=dict(color='purple', opacity=0.3)))
                # Add Buy Signals (with state-based logic)
                fig4.add_trace(go.Scatter(
                    x=buy_signal_dates,
                    y=buy_signal_prices,
                    mode='markers',
                    name='Buy Signal',
                    marker=dict(color='green', size=8, symbol='circle')
                ))

                # Add Sell Signals (with state-based logic)
                fig4.add_trace(go.Scatter(
                    x=sell_signal_dates,
                    y=sell_signal_prices,
                    mode='markers',
                    name='Sell Signal',
                    marker=dict(color='red', size=8, symbol='circle')
                ))
                # Highlight and annotate the last price
                last_price = stock_data['Close'].iloc[-1]
                last_date = stock_data.index[-1]
                fig4.add_trace(go.Scatter(x=[last_date], y=[last_price],
                                          mode='markers+text', name='Last Price',
                                          marker=dict(color='black', size=10),
                                          text=[f'{last_price:.2f}'], textposition="bottom right"))

                # Layout Customization
                fig4.update_layout(
                    title=f'EMA-Based Mean Reversion Strategy with RSI, MACD, and Volatility for {t}',
                    xaxis_title='Date',
                    yaxis_title='Close Price',
                    legend=dict(x=0, y=1),
                    template='plotly_white'
                )

                # Show the interactive Plotly chart in Streamlit
                st.plotly_chart(fig4)
#=======================================================================================================================
#               Arima Forecast
#=======================================================================================================================

                st.title(f"ARIMA Forecast {t}")

                forecast_steps = 365  # Number of days to forecast
                forecast_results = {}  # Dictionary to store each ticker's forecast DataFrame

                series = stock_data['Close']
                # Ensure the index has frequency (assume business days).
                if series.index.freq is None:
                    series = series.asfreq('B')

                try:
                    # Fit a SARIMAX model with seasonality.
                    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
                    model_fit = model.fit(disp=False)

                    # Get forecast for the specified number of steps.
                    forecast_result = model_fit.get_forecast(steps=forecast_steps)
                    forecast_mean = forecast_result.predicted_mean
                    conf_int = forecast_result.conf_int()

                    # Standard error of the forecast (used for probability calculations).
                    std_err = forecast_result.se_mean  # 1D array-like of standard errors
                except Exception as e:
                    print(f"SARIMAX forecast failed for ticker {ticker}: {e}")
                    continue

                # Create forecast index starting one day after the last observed date.
                last_date = series.index[-1]
                forecast_index = pd.date_range(
                    last_date + pd.Timedelta(days=1),
                    periods=forecast_steps,
                    freq='B'
                )

                # Build a forecast DataFrame.
                forecast_df = pd.DataFrame({
                    'Forecast': forecast_mean,
                    'Lower CI': conf_int.iloc[:, 0],
                    'Upper CI': conf_int.iloc[:, 1]
                }, index=forecast_index)

                # Rename columns to include the ticker symbol (e.g., "spy_forecast").
                forecast_df.columns = [
                    f"{ticker.lower()}_{col.replace(' ', '_').lower()}"
                    for col in forecast_df.columns
                ]

                # Example probability: Probability forecast will be above the last historical close
                current_price = series.iloc[-1]
                forecast_col = f"{ticker.lower()}_forecast"
                lower_ci_col = f"{ticker.lower()}_lower_ci"
                upper_ci_col = f"{ticker.lower()}_upper_ci"

                # Compute probability that the forecast is above current_price on each day
                # P(X > threshold) = 1 - CDF(threshold)
                prob_col = f"{ticker.lower()}_prob_above_current"
                forecast_df[prob_col] = 1 - norm.cdf(
                    x=current_price,
                    loc=forecast_df[forecast_col],
                    scale=std_err
                )

                # Store the forecast results in a dictionary
                forecast_results[ticker] = forecast_df

                # --------------------- Plotly Plot --------------------- #
                fig6 = go.Figure()

                # Plot historical data
                fig6.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series,
                        mode='lines',
                        name='Historical Close'
                    )
                )

                # Plot forecast
                fig6.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[forecast_col],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='red')
                    )
                )

                # Fill between lower and upper confidence intervals
                fig6.add_trace(
                    go.Scatter(
                        x=list(forecast_df.index) + list(forecast_df.index[::-1]),
                        y=list(forecast_df[upper_ci_col]) + list(forecast_df[lower_ci_col][::-1]),
                        fill='toself',
                        fillcolor='rgba(255, 192, 203, 0.3)',  # pinkish
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name='95% CI'
                    )
                )

                # Probability line on a secondary y-axis
                fig6.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df[prob_col],
                        mode='lines',
                        name='Prob(Above Current)',
                        line=dict(color='blue'),
                        yaxis='y2'
                    )
                )

                # --------------------- Annotations --------------------- #
                # Current (last historical) price
                fig6.add_annotation(
                    x=series.index[-1],
                    y=current_price,
                    text=f"Current: {current_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xanchor='left',
                    yanchor='bottom'
                )

                # Last forecasted price
                forecast_last_price = forecast_df[forecast_col].iloc[-1]
                forecast_last_date = forecast_df.index[-1]
                fig6.add_annotation(
                    x=forecast_last_date,
                    y=forecast_last_price,
                    text=f"Forecast End: {forecast_last_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xanchor='left',
                    yanchor='bottom'
                )

                # Top price in the upper CI
                top_price = forecast_df[upper_ci_col].max()
                top_date = forecast_df[upper_ci_col].idxmax()
                fig6.add_annotation(
                    x=top_date,
                    y=top_price,
                    text=f"Top: {top_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xanchor='left',
                    yanchor='bottom'
                )

                # Bottom price in the lower CI
                bottom_price = forecast_df[lower_ci_col].min()
                bottom_date = forecast_df[lower_ci_col].idxmin()
                fig6.add_annotation(
                    x=bottom_date,
                    y=bottom_price,
                    text=f"Bottom: {bottom_price:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    xanchor='left',
                    yanchor='top'
                )

                # --------------------- Figure Layout --------------------- #
                fig6.update_layout(
                    title=f"SARIMAX Forecast for {ticker}",
                    xaxis_title="Date",
                    yaxis_title="Close Price",
                    template="plotly_white",
                    legend=dict(x=0, y=1.05, orientation='h'),
                    yaxis2=dict(
                        title="Probability",
                        overlaying='y',
                        side='right',
                        range=[0, 1]  # Probability range
                    )
                )

                st.plotly_chart(fig6)

                stock_data['Cumulative Return'] = (1 + stock_data['Return']).cumprod() - 1

                # Append the processed data and its symbol to the list
                processed_data_list.append((stock_data, t))

        except Exception as e:
            st.error(f"An error occurred for {t}: {e}")

# Combine the data from all tickers if available
if processed_data_list:
    combined_data = combine_data(*processed_data_list)
    st.header("Combined Data")
    st.write(combined_data)
else:
    st.write("No data found")
