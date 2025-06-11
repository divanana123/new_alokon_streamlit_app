import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import timedelta

@st.cache_data
def load_data():
    data = pd.read_csv('data_alokon_preprocessed.csv')
    data['Periode'] = pd.to_datetime(data['Periode'])
    data['Nama Puskesmas Lower'] = data['Nama Puskesmas'].str.lower()
    return data

data = load_data()

st.title("📊 Analisis & Prediksi Alokon dengan LSTM & SVM")
nama_dicari = st.text_input("Masukkan nama Puskesmas:").strip().lower()

if nama_dicari:
    cocok = data[data['Nama Puskesmas Lower'].str.contains(nama_dicari)]
    if cocok.empty:
        st.warning(f"Tidak ditemukan puskesmas dengan nama mengandung '{nama_dicari}'")
    else:
        list_puskesmas_ditemukan = cocok['Nama Puskesmas'].unique()
        for puskesmas in list_puskesmas_ditemukan:
            st.subheader(f"🔍 Analisis: {puskesmas}")
            df_puskesmas = data[data['Nama Puskesmas'] == puskesmas]
            populer = df_puskesmas['Jenis Alokon'].value_counts().idxmax()
            st.markdown(f"💊 **Alat kontrasepsi paling populer:** {populer}")
            df_ts = df_puskesmas.groupby('Periode')['Jumlah Penggunaan'].sum().reset_index()
            penggunaan = df_ts['Jumlah Penggunaan'].values
            if len(df_ts) <= 6:
                st.warning("⚠️ Data tidak cukup untuk dianalisis.")
                continue
            tren = []
            for i in range(1, len(penggunaan)):
                delta = penggunaan[i] - penggunaan[i - 1]
                if delta > 2:
                    tren.append(1)
                elif delta < -2:
                    tren.append(-1)
                else:
                    tren.append(0)
            fitur = []
            for i in range(3, len(penggunaan)):
                diff1 = penggunaan[i - 1] - penggunaan[i - 2]
                diff2 = penggunaan[i - 2] - penggunaan[i - 3]
                diff3 = penggunaan[i] - penggunaan[i - 1]
                fitur.append([diff1, diff2, diff3])
            tren = tren[2:]
            if len(fitur) > 5:
                X_svm = np.array(fitur)
                y_svm = np.array(tren)
                X_train, X_test = X_svm[:-1], X_svm[-1:]
                y_train, y_test = y_svm[:-1], y_svm[-1:]
                scaler_svm = StandardScaler()
                X_train = scaler_svm.fit_transform(X_train)
                X_test = scaler_svm.transform(X_test)
                svm_model = SVC(kernel='linear')
                svm_model.fit(X_train, y_train)
                y_pred_svm = svm_model.predict(X_test)
                label_dict = {-1: 'Turun', 0: 'Stabil', 1: 'Naik'}
                hasil_tren = label_dict[y_pred_svm[0]]
                st.markdown(f"📈 **Tren penggunaan bulan terakhir:** {hasil_tren}")
            else:
                st.warning("⚠️ Data tidak cukup untuk klasifikasi tren dengan SVM.")
            scaler = MinMaxScaler()
            df_ts['Normalized'] = scaler.fit_transform(df_ts[['Jumlah Penggunaan']])
            def create_sequences(series, seq_length=6):
                x, y = [], []
                for i in range(len(series) - seq_length):
                    x.append(series[i:i + seq_length])
                    y.append(series[i + seq_length])
                return np.array(x), np.array(y)
            seq_length = 6
            X_lstm, y_lstm = create_sequences(df_ts['Normalized'].values, seq_length)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(seq_length, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_lstm, y_lstm, epochs=50, verbose=0)
            y_pred = model.predict(X_lstm)
            y_true = scaler.inverse_transform(y_lstm.reshape(-1, 1))
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            mae = mean_absolute_error(y_true, y_pred_rescaled)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_rescaled))
            r2 = r2_score(y_true, y_pred_rescaled)
            st.markdown(f"**MAE**: {mae:.2f} | **RMSE**: {rmse:.2f} | **R²**: {r2:.2f}")
            fig, ax = plt.subplots()
            ax.plot(df_ts['Periode'][seq_length:], y_true, label='Aktual')
            ax.plot(df_ts['Periode'][seq_length:], y_pred_rescaled, label='Prediksi')
            ax.set_title(f'Tren Jumlah Penggunaan Alokon - {puskesmas}')
            ax.set_xlabel('Periode')
            ax.set_ylabel('Jumlah Penggunaan')
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            next_sequence = df_ts['Normalized'].values[-seq_length:].tolist()
            predictions = []
            for _ in range(5):
                input_array = np.array(next_sequence[-seq_length:]).reshape((1, seq_length, 1))
                next_pred = model.predict(input_array, verbose=0)
                predictions.append(next_pred[0][0])
                next_sequence.append(next_pred[0][0])
            predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            last_date = df_ts['Periode'].max()
            future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(), periods=5, freq='MS')
            fig2, ax2 = plt.subplots()
            ax2.plot(df_ts['Periode'][seq_length:], y_true, label='Aktual')
            ax2.plot(df_ts['Periode'][seq_length:], y_pred_rescaled, label='Prediksi LSTM')
            ax2.plot(future_dates, predictions_rescaled, 'o--', label='Prediksi 5 Bulan Kedepan', color='orange')
            ax2.set_title(f'Prediksi 5 Bulan Kedepan - {puskesmas}')
            ax2.set_xlabel('Periode')
            ax2.set_ylabel('Jumlah Penggunaan')
            ax2.legend()
            ax2.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig2)
            st.markdown("📅 **Prediksi jumlah penggunaan untuk 5 bulan ke depan:**")
            for tgl, nilai in zip(future_dates, predictions_rescaled.flatten()):
                st.write(f"{tgl.strftime('%B %Y')}: {nilai:.0f}")
