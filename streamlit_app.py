# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# -------------------
# 1. PAGE CONFIG
# -------------------
st.set_page_config(page_title="Dashboard Prediksi Harga Cabai",
                   page_icon="🌶️",
                   layout="wide")

# -------------------
# 2. TITLE
# -------------------
st.title("🌶️ Dashboard Prediksi Harga Cabai di Jawa Timur")
st.markdown("Upload data, lakukan analisis, uji stasioneritas, dan buat model ARIMAX dengan variabel dummy hari besar keagamaan.")

# -------------------
# 3. SIDEBAR MENU
# -------------------
st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV atau Excel", 
    type=["csv", "xls", "xlsx"]
)

if uploaded_file:
    file_name = uploaded_file.name.lower()

    # Baca file sesuai ekstensi
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Format file tidak didukung!")
        st.stop()

    # Pastikan kolom 'date' bertipe datetime
    if 'Tanggal' in df.columns:
        df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
        df = df.sort_values('Tanggal')
    else:
        st.error("Kolom 'Tanggal' tidak ditemukan di dataset!")

    # -------------------
    # TAB MENU
    # -------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Deskripsi Data",
                                      "📈 Visualisasi",
                                      "🧪 Uji Stasioneritas",
                                      "🤖 Model ARIMAX"])

    # -------------------
    # TAB 1: DESKRIPSI
    # -------------------
    with tab1:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe())

        st.markdown("### Data Awal")
        st.dataframe(df.head())

    # -------------------
    # TAB 2: VISUALISASI
    # -------------------
    with tab2:
        st.subheader("Visualisasi Harga")
        plt.figure(figsize=(10, 5))
        sns.lineplot(x='Tanggal', y='Harga', data=df)
        plt.title("Pergerakan Harga Cabai")
        plt.xlabel("Tanggal")
        plt.ylabel("Harga")
        st.pyplot(plt)

    # -------------------
    # TAB 3: UJI STASIONERITAS
    # -------------------
    with tab3:
        st.subheader("Uji ADF (Augmented Dickey-Fuller)")
        adf_result = adfuller(df['Harga'])
        st.write(f"ADF Statistic : {adf_result[0]:.4f}")
        st.write(f"p-value       : {adf_result[1]:.4f}")
        if adf_result[1] < 0.05:
            st.success("Data stasioner (p-value < 0.05)")
        else:
            st.warning("Data belum stasioner (p-value >= 0.05)")

        st.markdown("### Plot ACF dan PACF")
        fig_acf, ax_acf = plt.subplots()
        plot_acf(df['Harga'], ax=ax_acf, lags=30)
        st.pyplot(fig_acf)

        fig_pacf, ax_pacf = plt.subplots()
        plot_pacf(df['Harga'], ax=ax_pacf, lags=30)
        st.pyplot(fig_pacf)

    # -------------------
    # TAB 4: MODEL ARIMAX
    # -------------------
    with tab4:
        st.subheader("Pemodelan ARIMAX")

        # Pilih variabel dummy hari besar
        exog_vars = [col for col in df.columns if col not in ['Tanggal', 'Harga']]
        if exog_vars:
            st.write("Variabel exogenous terdeteksi:", exog_vars)
            exog_data = df[exog_vars]
        else:
            st.warning("Tidak ada variabel dummy hari besar di dataset!")
            exog_data = None

        # Input parameter model
        p = st.number_input("p (AR)", 0, 10, 1)
        d = st.number_input("d (Difference)", 0, 2, 1)
        q = st.number_input("q (MA)", 0, 10, 1)

        if st.button("Fit Model"):
            try:
                model = ARIMA(df['Harga'], order=(p, d, q), exog=exog_data)
                model_fit = model.fit()
                st.write(model_fit.summary())

                # Forecast ke depan
                n_forecast = st.number_input("Jumlah hari prediksi", 1, 30, 7)
                forecast = model_fit.get_forecast(steps=n_forecast, exog=exog_data.tail(n_forecast) if exog_data is not None else None)
                forecast_df = forecast.summary_frame()

                st.subheader("Hasil Prediksi")
                st.dataframe(forecast_df)

                # Plot hasil prediksi
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['Tanggal'], df['Harga'], label="Data Aktual")
                ax.plot(pd.date_range(df['Tanggal'].iloc[-1], periods=n_forecast+1, freq='D')[1:], forecast_df['mean'], label="Forecast", color='red')
                ax.fill_between(pd.date_range(df['Tanggal'].iloc[-1], periods=n_forecast+1, freq='D')[1:],
                                forecast_df['mean_ci_lower'],
                                forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
                ax.set_title("Prediksi Harga Cabai")
                ax.set_xlabel("Tanggal")
                ax.set_ylabel("Harga")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
