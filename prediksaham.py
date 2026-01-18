# app.py
# Aplikasi Web Prediksi Saham
# Perbandingan Linear Regression (Basic) vs LSTM (Advanced)
# Data diubah ke format Sequence / Sliding Window
# Evaluasi menggunakan RMSE dan MAE (Forecasting)

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Prediksi Saham: LR vs LSTM", layout="wide")

st.title("📈 Aplikasi Web Prediksi Harga Saham")
st.subheader("Upload Dataset CSV (External Dataset)")

# =============================
# Upload Dataset (Custom: Data_Saham_GOTO.csv)
# =============================
uploaded_file = st.file_uploader("Upload file CSV (format seperti Data_Saham_GOTO.csv)", type=["csv"])

window_size = st.slider("Window Size (Sequence Length)", 5, 60, 20)

if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)

    # =============================
    # Cleaning Dataset GOTO
    # =============================
    # Baris 0 = header ticker, baris 1 = label Date
    data = raw_data.iloc[2:].copy()
    data.rename(columns={'Price': 'Date'}, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])

    numeric_cols = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data[['Date', 'Close']].dropna()
    data.set_index('Date', inplace=True)

    st.write("### Dataset Setelah Cleaning")
    st.dataframe(data.head())

    st.write("### Grafik Harga Saham (Close)")
    st.line_chart(data)

    # =============================
    # Sliding Window Function
    # =============================
    def create_sliding_window(series, window):
        X, y = [], []
        for i in range(len(series) - window):
            X.append(series[i:i+window])
            y.append(series[i+window])
        return np.array(X), np.array(y)

    # =============================
    # Preprocessing
    # =============================
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']].values)

    X, y = create_sliding_window(scaled_data, window_size)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # =============================
    # Linear Regression Model
    # =============================
    X_train_lr = X_train.reshape(X_train.shape[0], -1)
    X_test_lr = X_test.reshape(X_test.shape[0], -1)

    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train)

    y_pred_lr = lr_model.predict(X_test_lr)

    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    mae_lr = mean_absolute_error(y_test, y_pred_lr)

    # =============================
    # LSTM Model
    # =============================
    model = Sequential([
        LSTM(50, input_shape=(window_size, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred_lstm = model.predict(X_test)

    rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
    mae_lstm = mean_absolute_error(y_test, y_pred_lstm)

    # =============================
    # Inverse Scaling
    # =============================
    y_test_inv = scaler.inverse_transform(y_test)
    y_lr_inv = scaler.inverse_transform(y_pred_lr)
    y_lstm_inv = scaler.inverse_transform(y_pred_lstm)

    # =============================
    # Evaluation Results
    # =============================
    st.write("## 📊 Evaluasi Model (Forecasting)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### LSTM")
        st.metric("RMSE", f"{rmse_lr:.4f}")
        st.metric("MAE", f"{mae_lr:.4f}")

    with col2:
        st.write("### Linear Regression")
        st.metric("RMSE", f"{rmse_lstm:.4f}")
        st.metric("MAE", f"{mae_lstm:.4f}")

    # =============================
    # Visualization Prediction
    # =============================
    st.write("## 🔍 Perbandingan Prediksi")

    result_df = pd.DataFrame({
        "Actual": y_test_inv.flatten(),
        "LSTM": y_lr_inv.flatten(),
        "Linear Regression": y_lstm_inv.flatten()
    })

    st.line_chart(result_df)

    # =============================
    # Evaluation Results (Regression)
    # =============================
    st.write("## 📊 Evaluasi Model (Forecasting / Regresi)")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### LSTM")
        st.metric("RMSE", f"{rmse_lr:.4f}")
        st.metric("MAE", f"{mae_lr:.4f}")

    with col2:
        st.write("### Linear Regression")
        st.metric("RMSE", f"{rmse_lstm:.4f}")
        st.metric("MAE", f"{mae_lstm:.4f}")

    # =============================
    # F1-Score Evaluation (Directional Classification)
    # =============================
    st.write("## 🎯 Evaluasi F1-Score (Arah Harga Naik / Turun)")

    # Actual direction (1 = naik, 0 = turun)
    y_actual_dir = (y_test_inv[1:] > y_test_inv[:-1]).astype(int)

    # Predicted direction
    y_lr_dir = (y_lr_inv[1:] > y_test_inv[:-1]).astype(int)
    y_lstm_dir = (y_lstm_inv[1:] > y_test_inv[:-1]).astype(int)

    from sklearn.metrics import f1_score

    f1_lr = f1_score(y_actual_dir, y_lr_dir)
    f1_lstm = f1_score(y_actual_dir, y_lstm_dir)

    col1, col2 = st.columns(2)
    with col1:
        st.write("### LSTM")
        st.metric("F1-Score", f"{f1_lr:.4f}")

    with col2:
        st.write("### Linear Regression")
        st.metric("F1-Score", f"{f1_lstm:.4f}")

    # =============================
    # Visualization Prediction
    # =============================
    st.write("## 🔍 Perbandingan Prediksi")

    result_df = pd.DataFrame({
        "Actual": y_test_inv.flatten(),
        "LSTM": y_lr_inv.flatten(),
        "Linear Regression": y_lstm_inv.flatten()
    })

    st.line_chart(result_df)

    st.info("Evaluasi menggunakan RMSE & MAE untuk regresi, serta F1-Score untuk klasifikasi arah harga (naik/turun) jika data bersifat imbalance.")

else:
    st.warning("Silakan upload dataset CSV (contoh: Data_Saham_GOTO.csv).")("Silakan upload dataset CSV untuk memulai.")
