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
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV/Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Baca file
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    # Pastikan kolom tanggal ada dan jadi index
    data.columns = data.columns.str.strip()  # bersihkan spasi
    if 'Tanggal' in data.columns:
        data['Tanggal'] = pd.to_datetime(data['Tanggal'], errors='coerce')
        data.set_index('Tanggal', inplace=True)
    elif 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data.set_index('date', inplace=True)
    else:
        st.error("Kolom tanggal tidak ditemukan!")
        st.stop()

    # Tab Data
    tab_data, tab_visual, tab_arimax = st.tabs(["📊 Data", "📈 Visualisasi", "⚙ Model ARIMAX"])
  
    # -------------------
    # TAB 1: DATA
    # -------------------
    with tab_data:
        st.subheader("Data Awal")
        st.dataframe(data)

        st.subheader("Cek Missing Value")
        missing_values = data.isnull().sum()
        if missing_values.sum() == 0:
            st.success("Data tidak memiliki missing value")
        st.write(missing_values)

        st.subheader("Data Visualisasi")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data["Harga"])
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga")
        ax.set_title('Grafik Harga Cabai Keriting di Jawa Timur 2021-2024')
        ax.grid()
        st.pyplot(fig)

        st.subheader("Statistik Deskriptif Harga per Tahun")
        data["Tahun"] = data.index.to_period("Y")
        statistik = data.groupby("Tahun")["Harga"].describe()
        st.dataframe(statistik)

    # -------------------
    # TAB 2: UJI STASIONERITAS
    # -------------------
    with tab_stasioneritas:
        st.subheader("Uji Stasioneritas - Augmented Dickey-Fuller Test")
    
        # --- 1. Uji ADF Awal ---
        result = adfuller(data["Harga"].dropna())
        st.write("### Hasil Uji ADF (Data Asli)")
        st.write(f"Test Statistic: {result[0]:.6f}")
        st.write(f"p-value: {result[1]:.6f}")
        st.write(f"# Lags Used: {result[2]}")
        st.write(f"Number of Observations: {result[3]}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key}: {value:.6f}")
        if result[1] <= 0.05:
            st.success("Interpretasi: Data stasioner")
            data_diff = data["Harga"].diff().dropna()  # tetap buat differencing agar ada untuk ACF/PACF
        else:
            st.warning("Interpretasi: Data tidak stasioner, akan dilakukan differencing")
            # --- 2. Differencing ---
            data_diff = data["Harga"].diff().dropna()
            result_diff = adfuller(data_diff.dropna())
            st.write("### Hasil Uji ADF Setelah Differencing")
            st.write(f"Test Statistic: {result_diff[0]:.6f}")
            st.write(f"p-value: {result_diff[1]:.6f}")
            st.write(f"# Lags Used: {result_diff[2]}")
            st.write(f"Number of Observations: {result_diff[3]}")
            st.write("Critical Values:")
            for key, value in result_diff[4].items():
                st.write(f"   {key}: {value:.6f}")
            if result_diff[1] <= 0.05:
                st.success("Interpretasi: Data sudah stasioner setelah differencing")
            else:
                st.error("Interpretasi: Data masih belum stasioner setelah differencing")
    
        # --- 3. Plot ACF dan PACF ---
        st.subheader("Plot ACF & PACF (Setelah Differencing)")
        fig, axes = plt.subplots(1, 2, figsize=(16, 4))
        plot_acf(data_diff, lags=20, ax=axes[0])
        axes[0].set_title("ACF - Setelah Differencing")
        plot_pacf(data_diff, lags=20, ax=axes[1])
        axes[1].set_title("PACF - Setelah Differencing")
        st.pyplot(fig)

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
