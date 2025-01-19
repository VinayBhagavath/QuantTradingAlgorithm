import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical


def create_stock_dataframe(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    return df


def add_technical_features(df):
    # Bollinger Bands - 20 day range
    df['20DMA'] = df['Close'].rolling(window=20).mean()
    df['20DSD'] = df['Close'].rolling(window=20).std()
    df['BBUB'] = df['20DMA'] + (2 * df['20DSD'])
    df['BBLB'] = df['20DMA'] - (2 * df['20DSD'])

    # Relative Strength Index - RSI = 100 - (100/(1+(avgGain/avgLoss)))
    delta = df['Close'].diff(1)
    # only use value if the delta is positive
    # default to 0 gain if we had a loss
    df['gain'] = np.where(delta > 0, delta, 0)
    # only use value if the delta is negative
    df['loss'] = np.where(delta < 0, -delta, 0)
    df['avggain'] = df['gain'].rolling(window=14).mean()
    df['avgloss'] = df['loss'].rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (df['avggain'] / df['avgloss'])))

    # Add more moving averages with diff ranges
    df['50DMA'] = df['Close'].rolling(window=50).mean()
    df['10DMA'] = df['Close'].rolling(window=10).mean()

    # Moving Average Convergence Divergence - MACD = ExponentialMovingAverage12d-ExponentialMovingAverage26d
    # ewm is exponentially weighted moving average
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean(
    )-df['Close'].ewm(span=26, adjust=False).mean()
    df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Remove any null values, ex: delta of day 1 is None
    df = df.dropna(axis=0)

    # remove any unneeded columns
    df = df.drop('20DSD', axis=1)
    df = df.drop('gain', axis=1)
    df = df.drop('loss', axis=1)
    df = df.drop('avggain', axis=1)
    df = df.drop('avgloss', axis=1)

    return df


def preprocess_data(df, seq_len=60):
    # Feature Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.values)

    # Create sequences and labels
    sequences = []
    labels = []
    for i in range(seq_len, len(scaled_data) - 1):
        # Input sequence of length seq_len
        sequences.append(scaled_data[i-seq_len:i])
        # Label based on future price movement
        future_close = scaled_data[i + 1, df.columns.get_loc("Close")]
        current_close = scaled_data[i, df.columns.get_loc("Close")]
        if future_close > current_close:
            labels.append(1)  # Buy
        elif future_close < current_close:
            labels.append(-1)  # Sell
        else:
            labels.append(0)  # Hold

    return np.array(sequences), np.array(labels), scaler


def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    ticker = "PLTR"
    # yfinance API will round up to first trading day, use 1500 to ensure rounding up
    start_date = "1500-01-23"
    end_date = "2025-01-16"  # Arbitrary ending day in present
    df = create_stock_dataframe(ticker, start_date, end_date)
    df = add_technical_features(df)

    sequences, labels, scaler = preprocess_data(df, seq_len=60)

    labels = to_categorical([label + 1 for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42)

    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=16,
              validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    # also add backtesting for better eval
    print(f"Test Accuracy: {accuracy:.2f}")
