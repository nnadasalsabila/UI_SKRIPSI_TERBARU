# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
# 2. CSS KUSTOM
# -------------------
st.markdown("""
    <style>
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1e1e1e;
        }
        /* Tombol menu di sidebar */
        div[role="radiogroup"] > label {
            background-color: #d60000;
            color: white;
            border-radius: 10px;
            padding: 8px;
            margin: 5px 0px;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        div[role="radiogroup"] > label:hover {
            background-color: #a80000;
        }
        /* Kotak fitur & panduan */
        .red-box {
            background-color: #d60000;
            padding: 20px;
            border-radius: 15px;
            color: white;
            min-height: 250px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------
# 3. SIDEBAR MENU
# -------------------
menu = st.sidebar.radio("Navigasi", ["🏠 Homepage", "📊 Pemodelan & Prediksi"])

# -------------------
# 4. HALAMAN HOME
# -------------------
if menu == "🏠 Homepage":
    st.markdown("<h2 style='text-align: center;'>Dashboard Prediksi Harga Cabai di Jawa Timur</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>Website ini merupakan sistem prediksi harga komoditas cabai untuk membantu pemantauan fluktuasi harga cabai di Jawa Timur. Model prediksi yang digunakan adalah ARIMAX (Autoregressive Integrated Moving Average with Exogenous Variables).</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='red-box'><h4>Fitur:</h4><ul><li>Upload Data harga harian</li><li>Uji Stasioneritas</li><li>Model ARIMAX</li><li>Prediksi & Evaluasi</li></ul></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='red-box'><h4>Panduan Penggunaan:</h4><p>1. Pilih menu Pemodelan & Prediksi<br>2. Upload file CSV/Excel<br>3. Tentukan parameter model<br>4. Lihat hasil prediksi</p></div>", unsafe_allow_html=True)
# -------------------
# 5. HALAMAN ANALISIS
# -------------------
elif menu == "📊 Pemodelan & Prediksi":
  st.header("📂 Upload Data")
  uploaded_file = st.file_uploader("Upload file CSV/Excel", type=["csv", "xlsx"])

  # Buat tab hanya kalau data sudah ada
  tab_data, tab_stasioneritas, tab_splitting, tab_arima, tab_arimax, tab_predeval = st.tabs(["📊 Data", "📈 Uji Stasioneritas", "✂ Splitting Data", "⚙ Model ARIMA", "⚙ Model ARIMAX", "Prediksi & Evaluasi"])
    
  if uploaded_file:
      # Baca file
      if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
      else:
        data = pd.read_excel(uploaded_file)

      # Pastikan kolom tanggal ada dan jadi index
      data.columns = data.columns.str.strip()
      if 'Tanggal' in data.columns:
        data['Tanggal'] = pd.to_datetime(data['Tanggal'], errors='coerce')
        data.set_index('Tanggal', inplace=True)
      elif 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data.set_index('date', inplace=True)
      else:
        st.error("Kolom tanggal tidak ditemukan!")
        st.stop()
  else:
    with tab_data:
      st.info("Silahkan unggah file terlebih dahulu.")

  # ===== TAB DATA ===== #
  with tab_data:
      if 'data' in locals() and data is not None and not data.empty:
          st.subheader("Data Awal")
          st.dataframe(data)
  
          st.subheader("Cek Missing Value")
          missing_values = data.isnull().sum()
          if missing_values.sum() == 0:
              st.success("Data tidak memiliki missing value")
          else:
              st.write(missing_values)
  
          st.subheader("Data Visualisasi")
          fig, ax = plt.subplots(figsize=(12, 6))
          if "Harga" in data.columns:
              ax.plot(data.index, data["Harga"])
              ax.set_ylabel("Harga")
          else:
              st.warning("Kolom 'Harga' tidak ditemukan di data.")
          ax.set_xlabel("Tanggal")
          ax.set_title("Grafik Harga Cabai Keriting di Jawa Timur")
          ax.grid(True)
          st.pyplot(fig)
  
          st.subheader("Statistik Deskriptif Harga per Tahun")
          if "Harga" in data.columns:
              data_per_tahun = data.copy()
              data_per_tahun["Tahun"] = data_per_tahun.index.to_period("Y")
              statistik = data_per_tahun.groupby("Tahun")["Harga"].describe()
              st.dataframe(statistik)
          else:
              st.info("Statistik per tahun membutuhkan kolom 'Harga'.")
      else:
          pass
               
  # ===== TAB UJI STASIONERITAS ===== #
  with tab_stasioneritas:
      if 'data' in locals() and data is not None and not data.empty and "Harga" in data.columns:
          st.subheader("Uji Stasioneritas - Augmented Dickey-Fuller Test")
          # --- 1. Uji ADF Awal --- #
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
              data_diff = data["Harga"].diff().dropna()
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
      else:
          st.info("Silahkan unggah data terlebih dahulu untuk melakukan uji stasioneritas.")

  # ===== TAB SPLITTING DATA ===== #
  with tab_splitting:
      if 'data' in locals() and data is not None and not data.empty and all(col in data.columns for col in ["Harga", "Idul Adha", "Natal"]):
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
      else:
          st.info("Silahkan unggah data terlebih dahulu untuk melakukan splitting.")

  # ===== TAB PEMODELAN ARIMA ===== #
  with tab_arima:
    if uploaded_file is None:
        st.info("Silahkan unggah data terlebih dahulu untuk melakukan pemodelan ARIMA.")
    else:
        st.subheader("Pemodelan ARIMA")
        # Pastikan data sudah ada
        if 'data' in locals() and data is not None:
            data.index = pd.to_datetime(data.index)
    
            # Tentukan tanggal split
            split_date = '2024-12-26'
            y_train_arima = data['Harga'].loc[data.index < split_date]
            y_test_arima  = data['Harga'].loc[data.index >= split_date]
    
            st.write("Jumlah data y_train:", len(y_train_arima))
            st.write("Jumlah data y_test :", len(y_test_arima))
    
            st.header("🔍 Pencarian Model ARIMA Terbaik")
    
            from statsmodels.tsa.arima.model import ARIMA
            import pandas as pd
    
            # Input parameter range
            col1, col2 = st.columns(2)
            with col1:
                p_start = st.number_input("P mulai dari", min_value=0, max_value=10, value=0, key="arima_p_start")
                p_end = st.number_input("P sampai", min_value=p_start+1, max_value=10, value=5, key="arima_p_end")
            with col2:
                d_start = st.number_input("D mulai dari", min_value=0, max_value=4, value=1, key="arima_d_start")
                d_end = st.number_input("D sampai", min_value=d_start+1, max_value=4, value=2, key="arima_d_end")
            col3, col4 = st.columns(2)
            with col3:
                q_start = st.number_input("Q mulai dari", min_value=0, max_value=10, value=0, key="arima_q_start")
                q_end = st.number_input("Q sampai", min_value=q_start+1, max_value=10, value=8, key="arima_q_end")
    
            p = range(p_start, p_end)
            d = range(d_start, d_end)
            q = range(q_start, q_end)
    
            # Jalankan ARIMA
            if st.button("Jalankan ARIMA"):
                results = []
                significant_models = []
    
                with st.spinner("🔄 Sedang mencari kombinasi ARIMA terbaik..."):
                    for p_val in p:
                        for d_val in d:
                            for q_val in q:
                                try:
                                    model = ARIMA(y_train_arima, order=(p_val, d_val, q_val))
                                    model_fit = model.fit()
                                    pvalues = model_fit.pvalues
    
                                    if all(pv < 0.05 for pv in pvalues):
                                        significant_models.append({
                                            'Order (p,d,q)': (p_val, d_val, q_val),
                                            'AIC': model_fit.aic,
                                            'p-values': pvalues,
                                            'Summary': model_fit.summary(),
                                            'ModelFit': model_fit
                                        })
    
                                    results.append({
                                        'Order (p,d,q)': (p_val, d_val, q_val),
                                        'AIC': model_fit.aic,
                                        'p-values': pvalues,
                                        'Summary': model_fit.summary(),
                                        'ModelFit': model_fit
                                    })
    
                                except Exception as e:
                                    st.write(f"⚠️ Error pada ARIMA({p_val},{d_val},{q_val}): {e}")
                                    continue
    
                if significant_models:
                    summary_df = pd.DataFrame({
                        'Order (p,d,q)': [m['Order (p,d,q)'] for m in significant_models],
                        'AIC': [m['AIC'] for m in significant_models]
                    }).sort_values(by='AIC').reset_index(drop=True)
    
                    st.success("📌 Model yang signifikan berdasarkan p-value < 0.05:")
                    st.dataframe(summary_df)
                    best_model_info = min(significant_models, key=lambda x: x['AIC'])
                else:
                    st.warning("❌ Tidak ada model signifikan. Memilih model dengan AIC terkecil dari semua hasil.")
                    best_model_info = min(results, key=lambda x: x['AIC'])
    
                # Simpan model terbaik ke session_state
                st.session_state.arima_best_model = best_model_info['ModelFit']
                st.session_state.arima_best_order = best_model_info['Order (p,d,q)']
                st.session_state.arima_best_aic = best_model_info['AIC']
    
                st.markdown(f"**Model ARIMA terbaik:** {st.session_state.arima_best_order} (AIC: {st.session_state.arima_best_aic:.2f})")
    
                with st.expander("📄 Lihat Summary Model Terbaik"):
                    st.text(best_model_info['Summary'])
    
            # Tombol untuk uji diagnostik
            if st.button("Lakukan Uji Diagnostik"):
                if 'arima_best_model' in st.session_state:
                    from scipy import stats
                    from statsmodels.stats.diagnostic import acorr_ljungbox, het_goldfeldquandt
                    from statsmodels.tools.tools import add_constant
                    import numpy as np
    
                    x_dummy = np.arange(len(y_train_arima)).reshape(-1, 1)
                    x_dummy_const = add_constant(x_dummy)
    
                    residual_arima = st.session_state.arima_best_model.resid
    
                    # Uji KS
                    ks_stat, ks_p_value = stats.kstest(residual_arima, 'norm', args=(0, 1))
                    st.write(f"**Kolmogorov-Smirnov Test**")
                    st.write(f"Statistik KS: {ks_stat:.8f}")
                    st.write(f"P-value     : {ks_p_value:.8f}")
                    if ks_p_value > 0.05:
                        st.success("Residual terdistribusi normal (gagal menolak H0).")
                    else:
                        st.error("Residual tidak terdistribusi normal (menolak H0).")
    
                    # Uji White Noise - Ljung Box
                    ljung_box_result = acorr_ljungbox(residual_arima, lags=[10, 20, 30, 40], return_df=True)
                    st.write("**Ljung-Box Test**")
                    st.dataframe(ljung_box_result)
                    if (ljung_box_result['lb_pvalue'] > 0.05).all():
                        st.success("Residual adalah White Noise (gagal menolak H0).")
                    else:
                        st.error("Residual bukan White Noise (menolak H0).")
    
                    # Uji Goldfeld-Quandt
                    gq_test_arima = het_goldfeldquandt(residual_arima, x_dummy_const)
                    st.write("**Goldfeld-Quandt Test**")
                    st.write(f"Statistik GQ: {gq_test_arima[0]:.8f}")
                    st.write(f"P-value     : {gq_test_arima[1]:.8f}")
                    if gq_test_arima[1] <= 0.05:
                        st.error("Ada heteroskedastisitas (tolak H0).")
                    else:
                        st.success("Tidak ada heteroskedastisitas (gagal menolak H0).")
                else:
                    st.error("Model ARIMA belum dibuat. Jalankan ARIMA terlebih dahulu.")


  # ===== TAB PEMODELAN ARIMAX ===== #
  with tab_arimax:
    if uploaded_file is None:
        st.info("Silahkan unggah data terlebih dahulu untuk melakukan pemodelan ARIMAX.")
    else:
        # kode pemodelan ARIMAX di sini
        st.subheader("Pemodelan ARIMAX")
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import pandas as pd
    
        # === Input parameter dalam range ===
        col1, col2 = st.columns(2)
        with col1:
            p_start = st.number_input("P mulai dari", min_value=0, max_value=10, value=0, key="arimax_p_start")
            p_end = st.number_input("P sampai", min_value=p_start+1, max_value=10, value=5, key="arimax_p_end")
        with col2:
            d_start = st.number_input("D mulai dari", min_value=0, max_value=4, value=1, key="arimax_d_start")
            d_end = st.number_input("D sampai", min_value=d_start+1, max_value=4, value=2, key="arimax_d_end")
    
        col3, col4 = st.columns(2)
        with col3:
            q_start = st.number_input("Q mulai dari", min_value=0, max_value=10, value=0, key="arimax_q_start")
            q_end = st.number_input("Q sampai", min_value=q_start+1, max_value=10, value=7, key="arimax_q_end")
    
        # Buat range berdasarkan input
        p_range = range(p_start, p_end)
        d_range = range(d_start, d_end)
        q_range = range(q_start, q_end)
    
        # Inisialisasi agar tidak NameError di tab berikutnya
        best_model = None
        best_order = None
        best_aic = None
    
        if st.button("Jalankan ARIMAX"):
            results = []
            significant_models = []
    
            with st.spinner("Sedang melakukan fitting model ARIMAX..."):
                for p in p_range:
                    for d in d_range:
                        for q in q_range:
                            try:
                                model = SARIMAX(y_train, order=(p, d, q), exog=x_train)
                                model_fit = model.fit(disp=False)
                                pvalues = model_fit.pvalues
    
                                if all(pv < 0.05 for pv in pvalues):
                                    significant_models.append({
                                        'Order (p,d,q)': (p, d, q),
                                        'AIC': model_fit.aic,
                                        'p-values': pvalues,
                                        'Summary': model_fit.summary(),
                                        'ModelFit': model_fit
                                    })
    
                                results.append({
                                    'Order (p,d,q)': (p, d, q),
                                    'AIC': model_fit.aic,
                                    'p-values': pvalues,
                                    'Summary': model_fit.summary(),
                                    'ModelFit': model_fit
                                })
    
                            except Exception as e:
                                st.write(f"Error pada ARIMAX({p},{d},{q}): {e}")
                                continue
    
            # Pilih model terbaik
            if significant_models:
                summary_df = pd.DataFrame({
                    'Order (p,d,q)': [m['Order (p,d,q)'] for m in significant_models],
                    'AIC': [m['AIC'] for m in significant_models]
                }).sort_values(by='AIC').reset_index(drop=True)
    
                st.write("### Model signifikan (p-value < 0.05):")
                st.dataframe(summary_df)
                best_model_info = min(significant_models, key=lambda x: x['AIC'])
            else:
                st.warning("❌ Tidak ada model signifikan. Memilih model dengan AIC terkecil dari semua hasil.")
                best_model_info = min(results, key=lambda x: x['AIC'])
    
            arimax_best_model = best_model_info['ModelFit']
            arimax_best_order = best_model_info['Order (p,d,q)']
            arimax_best_aic = best_model_info['AIC']
    
            st.markdown(f"**Model ARIMAX terbaik:** {arimax_best_order} (AIC: {arimax_best_aic:.2f})")
    
            with st.expander("📄 Lihat Summary Model Terbaik"):
                st.text(best_model_info['Summary'])
    
            # -------------------
            # Uji Diagnostik
            # -------------------
            st.subheader("Uji Diagnostik Model Terbaik")
            from scipy import stats
            from statsmodels.stats.diagnostic import acorr_ljungbox, het_goldfeldquandt
            from statsmodels.tools.tools import add_constant
    
            residual = pd.DataFrame(arimax_best_model.resid)
    
            # Uji KS
            ks_stat, ks_p_value = stats.kstest(arimax_best_model.resid, 'norm', args=(0, 1))
            st.write(f"**Kolmogorov-Smirnov Test**")
            st.write(f"Statistik KS: {ks_stat:.8f}")
            st.write(f"P-value     : {ks_p_value:.8f}")
            if ks_p_value > 0.05:
                st.success("Residual terdistribusi normal (gagal menolak H0).")
            else:
                st.error("Residual tidak terdistribusi normal (menolak H0).")
    
            # Uji White Noise - Ljung Box
            ljung_box_result = acorr_ljungbox(residual, lags=[10, 20, 30, 40], return_df=True)
            st.write("**Ljung-Box Test**")
            st.dataframe(ljung_box_result)
            if (ljung_box_result['lb_pvalue'] > 0.05).all():
                st.success("Residual adalah White Noise (gagal menolak H0).")
            else:
                st.error("Residual bukan White Noise (menolak H0).")
    
            # Uji Heteroskedastisitas - Goldfeld Quandt
            x_train_const = add_constant(x_train)
            gq_test = het_goldfeldquandt(residual, x_train_const)
            st.write("**Goldfeld-Quandt Test**")
            st.write(f"Statistik GQ: {gq_test[0]:.8f}")
            st.write("P-value     :", gq_test[1])
            if gq_test[1] <= 0.05:
                st.error("Ada heteroskedastisitas (tolak H0)")
            else:
                st.success("Tidak ada heteroskedastisitas")
            
  # ===== TAB PREDIKSI & EVALUASI ===== #
  with tab_predeval:
    if uploaded_file is None:
        st.info("Silahkan unggah data terlebih dahulu untuk melakukan prediksi dan evaluasi.")
    else:
        # kode prediksi dan evaluasi di sini
        st.subheader("Prediksi & Evaluasi Model ARIMAX")
        if best_model is not None:
            st.success(f"Model terbaik: ARIMAX{best_order} dengan AIC = {best_aic:.2f}")
    
            # === 1. Prediksi ===
            pred_train = best_model.predict(
                start=y_train.index[0],
                end=y_train.index[-1],
                exog=x_train,
                dynamic=False
            )
    
            pred_test = best_model.predict(
                start=y_test.index[0],
                end=y_test.index[-1],
                exog=x_test,
                dynamic=False
            )
    
            # Gabungkan hasil prediksi
            prediksi = pd.concat([
                pred_train.rename('Prediksi_Train'),
                pred_test.rename('Prediksi_Test')
            ])
    
            # === 2. Visualisasi Prediksi Train + Test ===
            fig1, ax1 = plt.subplots(figsize=(18, 6))
            ax1.plot(y_train, label='Data Train (Aktual)', color='red')
            ax1.plot(y_test, label='Data Test (Aktual)', color='purple')
            ax1.plot(pred_train, label='Prediksi Train', color='lightskyblue')
            ax1.plot(pred_test, label='Prediksi Test', color='pink')
            ax1.axvline(y_test.index[0], color='green', linestyle='--', label='Train/Test Split')
            ax1.set_title(f'Prediksi Harga Cabai Keriting dengan ARIMAX {best_order}')
            ax1.set_xlabel('Tanggal')
            ax1.set_ylabel('Harga')
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
    
            # === 3. Visualisasi Prediksi Test Saja ===
            fig2, ax2 = plt.subplots(figsize=(12, 5))
            ax2.plot(y_test, label='Data Test (Aktual)', color='red')
            ax2.plot(pred_test, label='Prediksi Test', color='blue')
            ax2.set_title('Prediksi Data Test')
            ax2.set_xlabel('Tanggal')
            ax2.set_ylabel('Harga')
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig2)
    
            # === 4. Tabel Hasil Prediksi Test ===
            hasil_df = pd.DataFrame({
                'Tanggal': y_test.index,
                'Aktual': y_test.values,
                'Prediksi': pred_test.values
            })
            st.dataframe(hasil_df)
    
            # === 5. Evaluasi ===
            import numpy as np
            
            # Hitung MAPE manual
            def mean_absolute_percentage_error(y_true, y_pred):
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                mask = y_true != 0  # hindari pembagian nol
                return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            
            # === 5. Evaluasi ===
            mape_train = mean_absolute_percentage_error(y_train, pred_train)
            rmse_train = np.sqrt(np.mean((y_train - pred_train) ** 2))
            
            mape_test = mean_absolute_percentage_error(y_test, pred_test)
            rmse_test = np.sqrt(np.mean((y_test - pred_test) ** 2))
            
            st.markdown(f"""
            **Evaluasi Model ARIMAX {best_order}**
            - MAPE Train : {mape_train:.2f} %
            - RMSE Train : {rmse_train:.2f}
            - MAPE Test  : {mape_test:.2f} %
            - RMSE Test  : {rmse_test:.2f}
            """)
