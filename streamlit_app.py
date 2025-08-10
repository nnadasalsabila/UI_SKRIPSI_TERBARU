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
    tab_data, tab_stasioneritas, tab_splitting, tab_arimax = st.tabs(["📊 Data", "📈 Uji Stasioneritas", "✂ Splitting Data", "⚙ Model ARIMAX"])
  
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
    # TAB : SPLITTING DATA
    # -------------------
    with tab_splitting:
        st.subheader("Splitting Data - Training & Testing")
        # Buat DataFrame untuk ARIMAX
        data_arimax = pd.DataFrame({
            'Y': data['Harga'],
            'X1': data['Idul Adha'],
            'X2': data['Natal']
        })
    
        # Pastikan index adalah datetime
        data_arimax.index = pd.to_datetime(data_arimax.index)
    
        # Tentukan tanggal split
        split_date = '2024-12-25'
    
        # Split target (y)
        y_train = data_arimax['Y'].loc[data_arimax.index < split_date]
        y_test  = data_arimax['Y'].loc[data_arimax.index >= split_date]
    
        # Split eksogen (x)
        x_train = data_arimax[['X1', 'X2']].loc[data_arimax.index < split_date]
        x_test  = data_arimax[['X1', 'X2']].loc[data_arimax.index >= split_date]
    
        # Tampilkan hasil ke Streamlit
        st.write("Jumlah data y_train:", len(y_train))
        st.write("Jumlah data y_test :", len(y_test))
        st.write("Jumlah data x_train:", len(x_train))
        st.write("Jumlah data x_test :", len(x_test))
    
        st.write("📋 **Preview Data Training**")
        st.dataframe(y_train.head())
        st.dataframe(x_train.head())
        st.write("📋 **Preview Data Testing**")
        st.dataframe(y_test.head())
        st.dataframe(x_test.head())

    # -------------------
    # TAB : ARIMAX
    # -------------------

    with tab_arimax:
        st.subheader("Pemodelan ARIMAX")
    
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import pandas as pd
    
        # Range parameter ARIMAX
        p_values = range(0, 5)
        d_values = range(1, 2)  # biasanya 1 cukup
        q_values = range(0, 8)
    
        results = []
        significant_models = []
    
        with st.spinner("Sedang melakukan fitting model ARIMAX..."):
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = SARIMAX(y_train, order=(p, d, q), exog=x_train)
                            model_fit = model.fit(disp=False)
    
                            # Ambil p-value
                            pvalues = model_fit.pvalues
    
                            # Cek jika semua p-value < 0.05
                            if all(pv < 0.05 for pv in pvalues):
                                significant_models.append({
                                    'Order (p,d,q)': (p, d, q),
                                    'AIC': model_fit.aic,
                                    'p-values': pvalues,
                                    'Summary': model_fit.summary()
                                })
    
                            # Simpan semua hasil
                            results.append({
                                'Order (p,d,q)': (p, d, q),
                                'AIC': model_fit.aic,
                                'p-values': pvalues,
                                'Summary': model_fit.summary()
                            })
    
                        except Exception as e:
                            st.write(f"Error pada ARIMAX({p},{d},{q}): {e}")
                            continue
    
        if significant_models:
            # Dataframe model signifikan
            summary_df = pd.DataFrame({
                'Order (p,d,q)': [m['Order (p,d,q)'] for m in significant_models],
                'AIC': [m['AIC'] for m in significant_models]
            }).sort_values(by='AIC').reset_index(drop=True)
    
            st.write("### Model signifikan (p-value < 0.05):")
            st.dataframe(summary_df)
    
            # Ambil model terbaik (AIC terkecil)
            best_model = min(significant_models, key=lambda x: x['AIC'])
    
            st.write(f"### Model terbaik: ARIMAX{best_model['Order (p,d,q)']}")
            st.write(f"**AIC:** {best_model['AIC']}")
    
            # Tampilkan summary model terbaik
            st.text(best_model['Summary'])
    
        else:
            st.warning("Tidak ada model yang semua p-valunya < 0.05.")
