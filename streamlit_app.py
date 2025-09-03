 # streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.impute import SimpleImputer
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
            background-color: #5d0d02;
        }

        /* Ubah teks Navigasi */
        section[data-testid="stSidebar"] .stRadio > label:first-child {
            color: white !important;
            font-size: 24px !important;
            font-weight: bold !important;
            margin-top: -10px !important;
            margin-bottom: 15px !important;
            display: block;
        }

        /* Styling tombol radio */
        div[role="radiogroup"] > label {
            background-color: #fcfcfc;
            color: black; /* ubah warna teks tombol biar kontras */
            border-radius: 10px;
            padding: 8px;
            margin: 5px 0px;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        div[role="radiogroup"] > label:hover {
            background-color: #a61804;
            color: white !important; /* teks ikut putih saat hover */
        }

        /* Card besar */
        .big-card {
            background-color: white;
            color: black;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0px 6px 25px rgba(0, 0, 0, 0.25);
        }
        /* Container untuk dua kotak merah */
        .sub-section {
            display: flex;
            gap: 20px;
            margin-top: 20px;
        }
        /* Kotak merah masing-masing */
        .sub-col {
            flex: 1;
            background-color: #a61804;
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
        }
        .sub-col h4 {
            margin-top: 0;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------
# 2. SIDEBAR MENU
# -------------------
st.sidebar.markdown(
    "<h2 style='color: white; font-size: 24px; font-weight: bold; margin-bottom: 0px;'>Navigasi</h2>",
    unsafe_allow_html=True
)

menu = st.sidebar.radio("", ["🏠 Homepage", "📊 Pemodelan & Prediksi"])

# -------------------
# 3. HALAMAN HOME
# -------------------
if menu == "🏠 Homepage":
    st.markdown("""
        <div class='big-card'>
            <h2 style='text-align: center;'>Dashboard Prediksi Harga Cabai di Jawa Timur</h2>
            <p style='text-align: center;'>
                Website ini merupakan sistem prediksi harga komoditas cabai untuk membantu pemantauan fluktuasi harga cabai di Jawa Timur.
                Model prediksi yang digunakan adalah <b>ARIMAX</b> (Autoregressive Integrated Moving Average with Exogenous Variables).
            </p>
            <div class='sub-section'>
                <div class='sub-col'>
                    <h4>Fitur:</h4>
                    <p>
                        1. Analisis Data<br>
                        2. Uji Stasioneritas<br>
                        3. Splitting Data<br>
                        4. Pemodelan ARIMA<br>
                        5. Pemodelan ARIMAX<br>
                        6. Prediksi Mendatang<br>
                    </p>
                </div>
                <div class='sub-col'>
                    <h4>Syarat Penggunaan:</h4>
                    <p>
                        1. Dataset yang digunakan merupakan file CSV/Excel<br>
                        2. Dataset berisi kolom 'Tanggal', 'Harga' untuk variabel target, dan 'Idul Adha' 'Natal' untuk variabel eksogen'<br>
                        3. Pengguna wajib menginputkan parameter secara manual<br>
                        4. Hasil prediksi yang ditampilkan hanya untuk periode satu minggu mendatang<br>
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -------------------
# 4. HALAMAN ANALISIS
# -------------------
elif menu == "📊 Pemodelan & Prediksi":
  st.header("📂 Upload Data")
  uploaded_file = st.file_uploader("Upload file CSV/Excel", type=["csv", "xlsx"])

  # Buat tab hanya kalau data sudah ada
  tab_data, tab_stasioneritas, tab_splitting, tab_arima, tab_arimax, tab_predeval = st.tabs(["📊 Data", "📈 Uji Stasioneritas", "✂ Splitting Data", "⚙ Model ARIMA", "⚙ Model ARIMAX", "🎯 Prediksi Mendatang"])
    
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
  # ===== Fungsi untuk membuat dummy berdasarkan kalender =====
  def buat_dummy(Tanggal, hari_h):
      return ((Tanggal >= hari_h - pd.Timedelta(days=7)) &
              (Tanggal <= hari_h + pd.Timedelta(days=7))).astype(int)
  
  with tab_data:
      if 'data' in locals() and data is not None and not data.empty:
          st.subheader("Data Awal")
          st.dataframe(data)
  
          st.subheader("Cek Missing Value")
          missing_values = data.isnull().sum()
  
          # Buat DataFrame dari hasil missing value
          missing_df = pd.DataFrame({
              "Kolom": missing_values.index,
              "Jumlah Missing": missing_values.values
          })
          st.dataframe(missing_df, use_container_width=True)
  
          if missing_values.sum() == 0:
              st.success("Data tidak memiliki missing value")
          else:
              st.warning("Data memiliki missing value")
  
              # Tombol untuk melakukan imputasi khusus
              if st.button("Lakukan Imputasi (Median + Kalender)"):
                  data_imputed = data.copy()
  
                  # ===== 1. Imputasi Harga dengan Median =====
                  if "Harga" in data_imputed.columns:
                      imputer = SimpleImputer(strategy="mean")
                      data_imputed["Harga"] = imputer.fit_transform(data_imputed[["Harga"]])
  
                  # Pastikan kolom tanggal ada dan bertipe datetime
                  if "Tanggal" in data_imputed.columns:
                      data_imputed["Tanggal"] = pd.to_datetime(data_imputed["Tanggal"])
  
                      # ===== 2. Imputasi Dummy Berdasarkan Kalender =====
                      # Definisi tanggal perayaan (contoh tahun 2021)
                      idul_adha = pd.to_datetime("2021-07-20")
                      natal = pd.to_datetime("2021-12-25")
                      tahun_baru = pd.to_datetime("2021-01-01")
                   
                      # Drop dulu kolom lama biar nggak tercampur dengan None
                      for kolom in ["Idul Adha", "Natal", "Tahun Baru"]:
                          if kolom in data_imputed.columns:
                              data_imputed.drop(columns=[kolom], inplace=True)
  
                      # Overwrite penuh kolom dummy sesuai nama aslinya
                      data_imputed["Idul Adha"] = buat_dummy(data_imputed["Tanggal"], idul_adha)
                      data_imputed["Natal"] = buat_dummy(data_imputed["Tanggal"], natal)
                      data_imputed["Tahun Baru"] = buat_dummy(data_imputed["Tanggal"], tahun_baru)
  
                  # ===== Hasil Akhir =====
                  st.subheader("Data Setelah Imputasi")
                  st.dataframe(data_imputed, use_container_width=True)
                  st.success("Missing value berhasil ditangani dengan Median (Harga) dan Kalender (dummy).")

          st.subheader("📊 Visualisasi Harga Cabai")
          if "Harga" in data.columns:
              st.line_chart(data['Harga'])
          else:
              st.warning("Kolom 'Harga' tidak ditemukan di data.")
  
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
  
          # Inisialisasi state
          if "adf_awal_done" not in st.session_state:
              st.session_state.adf_awal_done = False
          if "adf_awal_result" not in st.session_state:
              st.session_state.adf_awal_result = None
          if "adf_diff_done" not in st.session_state:
              st.session_state.adf_diff_done = False
          if "adf_diff_result" not in st.session_state:
              st.session_state.adf_diff_result = None
          if "data_diff" not in st.session_state:
              st.session_state.data_diff = None
  
          # Step 1 - Uji ADF awal
          if st.button("Jalankan Uji ADF Awal"):
              result = adfuller(data["Harga"].dropna())
              st.session_state.adf_awal_result = result
              st.session_state.adf_awal_done = True
  
          # Tampilkan hasil uji ADF awal jika sudah dijalankan
          if st.session_state.adf_awal_done and st.session_state.adf_awal_result is not None:
              result = st.session_state.adf_awal_result
              st.write("### Hasil Uji ADF (Data Asli)")
              st.write(f"Test Statistic: {result[0]:.6f}")
              st.write(f"p-value: {result[1]:.6f}")
              st.write(f"Number of Observations: {result[3]}")
              st.write("Critical Values:")
              for key, value in result[4].items():
                  st.write(f"   {key}: {value:.6f}")
  
              if result[1] <= 0.05:
                  st.success("Interpretasi: Data stasioner")
              else:
                  st.warning("Interpretasi: Data tidak stasioner, lakukan differencing")
  
                  # Step 2 - Differencing + langsung Uji ADF kedua
                  if st.button("Lakukan Differencing & Uji ADF Kembali"):
                      st.session_state.data_diff = data["Harga"].diff().dropna()
                      st.session_state.adf_diff_result = adfuller(st.session_state.data_diff.dropna())
                      st.session_state.adf_diff_done = True
  
          # Tampilkan hasil ADF setelah differencing (persisten meskipun pindah tab)
          if st.session_state.adf_diff_done and st.session_state.adf_diff_result is not None:
              result_diff = st.session_state.adf_diff_result
              st.write("### Hasil Uji ADF Setelah Differencing")
              st.write(f"Test Statistic: {result_diff[0]:.6f}")
              st.write(f"p-value: {result_diff[1]:.6f}")
              st.write(f"Number of Observations: {result_diff[3]}")
              st.write("Critical Values:")
              for key, value in result_diff[4].items():
                  st.write(f"   {key}: {value:.6f}")
  
              if result_diff[1] <= 0.05:
                  st.success("Interpretasi: Data sudah stasioner setelah differencing")
              else:
                  st.error("Interpretasi: Data masih belum stasioner setelah differencing")
  
              # Plot ACF & PACF setelah differencing
              st.subheader("Plot ACF & PACF (Setelah Differencing)")
              fig, axes = plt.subplots(1, 2, figsize=(16, 4))
              plot_acf(st.session_state.data_diff, lags=20, ax=axes[0])
              axes[0].set_title("ACF - Setelah Differencing")
              plot_pacf(st.session_state.data_diff, lags=20, ax=axes[1])
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
  
          if 'data' in locals() and data is not None:
              data.index = pd.to_datetime(data.index)
  
              split_date = '2024-12-26'
              y_train_arima = data['Harga'].loc[data.index < split_date]
              y_test_arima = data['Harga'].loc[data.index >= split_date]
  
              st.header("🔍 Pencarian Model ARIMA Terbaik")
  
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
                                    
                      best_model_info = min(significant_models, key=lambda x: x['AIC'])
                  else:
                      st.warning("❌ Tidak ada model signifikan. Memilih model dengan AIC terkecil dari semua hasil.")
                      best_model_info = min(results, key=lambda x: x['AIC'])
                      summary_df = pd.DataFrame()
  
                  # Simpan semua hasil ke session_state
                  st.session_state.arima_best_model = best_model_info['ModelFit']
                  st.session_state.arima_best_order = best_model_info['Order (p,d,q)']
                  st.session_state.arima_best_aic = best_model_info['AIC']
                  st.session_state.arima_best_summary = best_model_info['Summary']
                  st.session_state.arima_results_df = summary_df
  
              # ==== TAMPILKAN HASIL JIKA SUDAH ADA DI SESSION_STATE ====
              if 'arima_best_model' in st.session_state:
  
                  if not st.session_state.arima_results_df.empty:
                      st.success("📌 Model yang signifikan berdasarkan p-value < 0.05:")
                      st.markdown(f"**Model ARIMA terbaik:** {st.session_state.arima_best_order} (AIC: {st.session_state.arima_best_aic:.2f})")
                      st.dataframe(st.session_state.arima_results_df)
  
                  with st.expander("📄 Lihat Summary Model Terbaik"):
                      st.text(st.session_state.arima_best_summary)
  
                  # UJI DIAGNOSTIK
                  if st.button("Lakukan Uji Diagnostik"):
                      from scipy import stats
                      from statsmodels.stats.diagnostic import acorr_ljungbox, het_goldfeldquandt
                      from statsmodels.tools.tools import add_constant
                      import numpy as np
  
                      x_dummy = np.arange(len(y_train_arima)).reshape(-1, 1)
                      x_dummy_const = add_constant(x_dummy)
                  
                      residual_arima = st.session_state.arima_best_model.resid
                  
                      # Uji KS
                      ks_stat, ks_p_value = stats.kstest(residual_arima, 'norm', args=(0, 1))
                  
                      # Uji Ljung-Box
                      ljung_box_result = acorr_ljungbox(residual_arima, lags=[10, 20, 30, 40], return_df=True)
                  
                      # Uji Goldfeld-Quandt
                      gq_test_arima = het_goldfeldquandt(residual_arima, x_dummy_const)
                  
                      # Simpan ke session_state
                      st.session_state["diagnostic_results"] = {
                          "ks": (ks_stat, ks_p_value),
                          "ljungbox": ljung_box_result,
                          "gq": gq_test_arima
                      }
                  
                  # Tampilkan hasil diagnostik jika sudah ada
                  if "diagnostic_results" in st.session_state:
                      st.subheader("🧪 Hasil Uji Diagnostik")
                  
                      ks_stat, ks_p_value = st.session_state["diagnostic_results"]["ks"]
                      st.write(f"**Kolmogorov-Smirnov Test**")
                      st.write(f"Statistik KS: {ks_stat:.8f}")
                      st.write(f"P-value     : {ks_p_value:.8f}")
                      if ks_p_value > 0.05:
                          st.success("Residual terdistribusi normal (gagal menolak H0).")
                      else:
                          st.error("Residual tidak terdistribusi normal (menolak H0).")
                  
                      ljung_box_result = st.session_state["diagnostic_results"]["ljungbox"]
                      st.write("**Ljung-Box Test**")
                      st.dataframe(ljung_box_result)
                      if (ljung_box_result['lb_pvalue'] > 0.05).all():
                          st.success("Residual adalah White Noise (gagal menolak H0).")
                      else:
                          st.error("Residual bukan White Noise (menolak H0).")
                  
                      gq_test_arima = st.session_state["diagnostic_results"]["gq"]
                      st.write("**Goldfeld-Quandt Test**")
                      st.write(f"Statistik GQ: {gq_test_arima[0]:.8f}")
                      st.write(f"P-value     : {gq_test_arima[1]:.8f}")
                      if gq_test_arima[1] <= 0.05:
                          st.error("Ada heteroskedastisitas (tolak H0).")
                      else:
                          st.success("Tidak ada heteroskedastisitas (gagal menolak H0).")

                  # === EVALUASI MAPE ARIMA ===
                  if st.button("Lakukan Prediksi & Evaluasi"):
                      import numpy as np
                  
                      # Fungsi MAPE manual
                      def mean_absolute_percentage_error(y_true, y_pred):
                          y_true, y_pred = np.array(y_true), np.array(y_pred)
                          mask = y_true != 0  # hindari pembagian nol
                          return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                  
                      # ===============================
                      # 1. Split ulang dengan new_split_date
                      # ===============================
                      new_split_date = '2024-12-25'
                      data.index = pd.to_datetime(data.index)
                      train_arima_new = data['Harga'].loc[data.index < new_split_date]
                      test_arima_new = data['Harga'].loc[data.index >= new_split_date]
                  
                      # ===============================
                      # 2. Prediksi untuk data train & test
                      # ===============================
                      pred_train = st.session_state.arima_best_model.predict(
                          start=train_arima_new.index[0],
                          end=train_arima_new.index[-1],
                          dynamic=False
                      )
                      pred_test = st.session_state.arima_best_model.predict(
                          start=test_arima_new.index[0],
                          end=test_arima_new.index[-1],
                          dynamic=False
                      )
                  
                      # ===============================
                      # 3. Hitung MAPE
                      # ===============================
                      mape_arima_train = mean_absolute_percentage_error(train_arima_new, pred_train)
                      mape_arima_test = mean_absolute_percentage_error(test_arima_new, pred_test)
                  
                      # Simpan ke session_state
                      st.session_state.mape_arima_train = mape_arima_train
                      st.session_state.mape_arima_test = mape_arima_test
                      st.session_state.pred_test_arima = pred_test
                      st.session_state.test_arima_new = test_arima_new
                  
                      # ===============================
                      # 4. Simpan hasil dataframe
                      # ===============================
                      hasil_df = pd.DataFrame({
                          'Tanggal': test_arima_new.index,
                          'Aktual': test_arima_new.values,
                          'Prediksi': pred_test.values
                      })
                      st.session_state.hasil_df_arima = hasil_df
                  
                  # ===============================
                  # TAMPILKAN HASIL JIKA SUDAH ADA
                  # ===============================
                  if "mape_arima_train" in st.session_state and "mape_arima_test" in st.session_state:
                      st.subheader("📊 Hasil Evaluasi Model")
                      st.write(f"**MAPE Train:** {st.session_state.mape_arima_train:.2f}%")
                      st.write(f"**MAPE Test :** {st.session_state.mape_arima_test:.2f}%")
                  
                      st.subheader("📉 Visualisasi Prediksi pada Data Test")
                      fig, ax = plt.subplots(figsize=(12, 5))
                      ax.plot(st.session_state.test_arima_new, label='Data Test (Aktual)', color='red')
                      ax.plot(st.session_state.pred_test_arima, label='Prediksi Test', color='blue')
                      ax.set_title('Prediksi Data Test')
                      ax.set_xlabel('Tanggal')
                      ax.set_ylabel('Harga')
                      ax.legend()
                      ax.grid(True)
                      st.pyplot(fig)
                  
                      # Tampilkan DataFrame Hasil Prediksi
                      st.dataframe(st.session_state.hasil_df_arima)

            
  # ===== TAB PEMODELAN ARIMAX ===== #
  with tab_arimax:
      if uploaded_file is None:
          st.info("Silahkan unggah data terlebih dahulu untuk melakukan pemodelan ARIMAX.")
      else:
          # kode pemodelan ARIMAX
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
  
          # Inisialisasi session_state
          if "arimax_best_model" not in st.session_state:
              st.session_state["arimax_best_model"] = None
              st.session_state["arimax_best_order"] = None
              st.session_state["arimax_best_aic"] = None
  
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
  
              # Simpan hasil model terbaik ke session_state
              st.session_state["arimax_best_model"] = best_model_info['ModelFit']
              st.session_state["arimax_best_order"] = best_model_info['Order (p,d,q)']
              st.session_state["arimax_best_aic"] = best_model_info['AIC']
              st.session_state["arimax_summary"] = best_model_info['Summary']
  
          # =========================
          # TAMPILKAN HASIL ARIMAX
          # =========================
          if st.session_state["arimax_best_model"] is not None:
              st.markdown(f"**Model ARIMAX terbaik:** {st.session_state['arimax_best_order']} (AIC: {st.session_state['arimax_best_aic']:.2f})")
  
              with st.expander("📄 Lihat Summary Model Terbaik"):
                  st.text(st.session_state["arimax_summary"])
  
              # -------------------
              # Uji Diagnostik
              # -------------------
              st.subheader("Uji Diagnostik Model Terbaik")
              from scipy import stats
              from statsmodels.stats.diagnostic import acorr_ljungbox, het_goldfeldquandt
              from statsmodels.tools.tools import add_constant
  
              if st.button("🔎 Jalankan Uji Diagnostik"):
                  residual = pd.DataFrame(st.session_state["arimax_best_model"].resid)
  
                  # Uji KS
                  ks_stat, ks_p_value = stats.kstest(st.session_state["arimax_best_model"].resid, 'norm', args=(0, 1))
  
                  # Uji White Noise - Ljung Box
                  ljung_box_result = acorr_ljungbox(residual, lags=[10, 20, 30, 40], return_df=True)
  
                  # Uji Heteroskedastisitas - Goldfeld Quandt
                  x_train_const = add_constant(x_train)
                  gq_test = het_goldfeldquandt(residual, x_train_const)
  
                  # Simpan hasil diagnostik
                  st.session_state['arimax_diagnostics'] = {
                      "KS": {"stat": ks_stat, "pvalue": ks_p_value},
                      "LjungBox": ljung_box_result,
                      "GoldfeldQuandt": {"stat": gq_test[0], "pvalue": gq_test[1]}
                  }
  
              # Jika sudah ada hasil diagnostik → tampilkan
              if "arimax_diagnostics" in st.session_state:
                  diag = st.session_state['arimax_diagnostics']
  
                  st.write("**Kolmogorov-Smirnov Test**")
                  st.write(f"Statistik KS: {diag['KS']['stat']:.8f}")
                  st.write(f"P-value     : {diag['KS']['pvalue']:.8f}")
                  if diag['KS']['pvalue'] > 0.05:
                      st.success("Residual terdistribusi normal (gagal menolak H0).")
                  else:
                      st.error("Residual tidak terdistribusi normal (menolak H0).")
  
                  st.write("**Ljung-Box Test**")
                  st.dataframe(diag['LjungBox'])
                  if (diag['LjungBox']['lb_pvalue'] > 0.05).all():
                      st.success("Residual adalah White Noise (gagal menolak H0).")
                  else:
                      st.error("Residual bukan White Noise (menolak H0).")
  
                  st.write("**Goldfeld-Quandt Test**")
                  st.write(f"Statistik GQ: {diag['GoldfeldQuandt']['stat']:.8f}")
                  st.write(f"P-value     : {diag['GoldfeldQuandt']['pvalue']:.8f}")
                  if diag['GoldfeldQuandt']['pvalue'] <= 0.05:
                      st.error("Ada heteroskedastisitas (tolak H0)")
                  else:
                      st.success("Tidak ada heteroskedastisitas")
  
              # === EVALUASI MAPE ARIMAX ===
              if st.button("Lakukan Evaluasi (MAPE)"):
                  import numpy as np
              
                  def mean_absolute_percentage_error(y_true, y_pred):
                      y_true, y_pred = np.array(y_true), np.array(y_pred)
                      mask = y_true != 0
                      return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
              
                  # ===============================
                  # 1. Prediksi Train & Test
                  # ===============================
                  pred_train = st.session_state["arimax_best_model"].predict(
                      start=y_train.index[0],
                      end=y_train.index[-1],
                      dynamic=False,
                      exog=x_train
                  )
              
                  pred_test = st.session_state["arimax_best_model"].predict(
                      start=y_test.index[0],
                      end=y_test.index[-1],
                      dynamic=False,
                      exog=x_test
                  )
              
                  # ===============================
                  # 2. Hitung MAPE
                  # ===============================
                  mape_train = mean_absolute_percentage_error(y_train, pred_train)
                  mape_test = mean_absolute_percentage_error(y_test, pred_test)
              
                  # Simpan ke session_state
                  st.session_state["mape_arimax_train"] = mape_train
                  st.session_state["mape_arimax_test"] = mape_test
                  st.session_state["pred_test_arimax"] = pred_test
                  st.session_state["pred_train_arimax"] = pred_train
                  st.session_state["y_test_arimax"] = y_test
              
                  # ===============================
                  # 3. Visualisasi Prediksi Test
                  # ===============================
                  st.subheader("📉 Visualisasi Prediksi pada Data Test")
                  fig, ax = plt.subplots(figsize=(12, 5))
                  ax.plot(y_test, label='Data Test (Aktual)', color='red')
                  ax.plot(pred_test, label='Prediksi Test', color='blue')
                  ax.set_title('Prediksi Data Test (ARIMAX)')
                  ax.set_xlabel('Tanggal')
                  ax.set_ylabel('Harga')
                  ax.legend()
                  ax.grid(True)
                  st.pyplot(fig)
              
                  # ===============================
                  # 4. DataFrame Hasil Prediksi
                  # ===============================
                  hasil_df = pd.DataFrame({
                      'Tanggal': y_test.index,
                      'Aktual': y_test.values,
                      'Prediksi': pred_test.values
                  })
                  st.session_state["hasil_df_arimax"] = hasil_df
                  st.dataframe(hasil_df)
              
              # ===============================
              # Tampilkan Hasil Evaluasi Jika Sudah Ada
              # ===============================
              if "mape_arimax_train" in st.session_state and "mape_arimax_test" in st.session_state:
                  st.subheader("📊 Hasil Evaluasi Model")
                  st.write(f"**MAPE Train:** {st.session_state['mape_arimax_train']:.2f}%")
                  st.write(f"**MAPE Test :** {st.session_state['mape_arimax_test']:.2f}%")
                
  # ===== TAB PREDIKSI MENDATANG ===== #
  with tab_predeval:
      st.header("📊 Prediksi & Evaluasi Model")
  
      # =========================
      # 1. Buat Eksogen Masa Depan
      # =========================
      st.subheader("🔧 Buat Eksogen Masa Depan")
  
      if st.button("Buat Eksogen Masa Depan"):
          n_forecast = 8
          future_dates = pd.date_range(start='2024-12-31', periods=n_forecast, freq='D')
  
          # Daftar tanggal Idul Adha
          idul_adha_dates = [
              pd.Timestamp('2022-07-10'),
              pd.Timestamp('2023-06-29'),
              pd.Timestamp('2024-06-17'),
              pd.Timestamp('2025-06-06')
          ]
  
          # Daftar tanggal Natal
          natal_dates = [
              pd.Timestamp('2022-12-25'),
              pd.Timestamp('2023-12-25'),
              pd.Timestamp('2024-12-25'),
              pd.Timestamp('2025-12-25')
          ]
  
          # Range Idul Adha ±7 hari
          idul_adha_ranges = set()
          for d in idul_adha_dates:
              start_range = d - pd.Timedelta(days=7)
              end_range = d + pd.Timedelta(days=7)
              idul_adha_ranges.update(pd.date_range(start=start_range, end=end_range))
  
          # Range Natal ±7 hari
          natal_ranges = set()
          for d in natal_dates:
              start_range = d - pd.Timedelta(days=7)
              end_range = d + pd.Timedelta(days=7)
              natal_ranges.update(pd.date_range(start=start_range, end=end_range))
  
          # Exogenous future
          future_exog = pd.DataFrame({
              'X1': [1 if date in idul_adha_ranges else 0 for date in future_dates],
              'X2': [1 if date in natal_ranges else 0 for date in future_dates]
          }, index=future_dates)
  
          # Simpan di session_state
          st.session_state["future_dates"] = future_dates
          st.session_state["future_exog"] = future_exog
          st.success("✅ Eksogen masa depan berhasil dibuat!")
  
      # Tampilkan eksogen jika sudah ada (tetap tampil meskipun pindah tab/proses lain)
      if "future_exog" in st.session_state:
          st.subheader("📋 Tabel Eksogen Masa Depan")
          st.dataframe(st.session_state["future_exog"])
  
      # =========================
      # 2. Forecast ARIMAX
      # =========================
      if st.button("Prediksi Mendatang (Forecast ARIMAX)"):
          if "future_exog" not in st.session_state:
              st.error("❌ Harap buat eksogen masa depan terlebih dahulu!")
          else:
              n_forecast = len(st.session_state["future_exog"])
              future_dates = st.session_state["future_dates"]
              future_exog = st.session_state["future_exog"]
  
              # Forecast dengan best model ARIMAX
              forecast = st.session_state["arimax_best_model"].forecast(
                  steps=n_forecast, exog=future_exog
              )
              forecast.index = future_dates
  
              # Simpan hasil forecast
              forecast_df = pd.DataFrame({
                  "Tanggal": forecast.index,
                  "Prediksi_ARIMAX": forecast.values
              })
              st.session_state["forecast_arimax"] = forecast
              st.session_state["forecast_df_arimax"] = forecast_df
              st.success("✅ Forecast masa depan berhasil dibuat!")
  
      # Tampilkan hasil forecast jika sudah ada
      if "forecast_df_arimax" in st.session_state:
          st.subheader("📋 Tabel Hasil Prediksi Mendatang")
          st.dataframe(st.session_state["forecast_df_arimax"])
  
          # === Visualisasi ===
          fig, ax = plt.subplots(figsize=(15, 6))
  
          # Data aktual gabungan
          ax.plot(pd.concat([y_train, y_test]),
                  label='Data Aktual (Train & Test)', color='black')
  
          # Prediksi train (ambil dari session_state)
          if "pred_train_arimax" in st.session_state:
              ax.plot(st.session_state["pred_train_arimax"]
                      .loc[st.session_state["pred_train_arimax"].index >= '2024-12-01'],
                      label='Prediksi Train', color='blue')
  
          # Prediksi test
          if "pred_test_arimax" in st.session_state:
              ax.plot(st.session_state["pred_test_arimax"],
                      label='Prediksi Test', color='red')
  
          # Forecast
          ax.plot(st.session_state["forecast_arimax"],
                  label='Forecast 7 Hari ke Depan', color='green')
  
          # Garis pemisah
          ax.axvline(y_test.index[0], color='orange', linestyle='--', label='Train/Test Split')
          ax.axvline(st.session_state["forecast_arimax"].index[0],
                     color='purple', linestyle='--', label='Awal Forecast Masa Depan')
  
          ax.set_title('Prediksi Harga Cabai Keriting (Train, Test, dan Forecast Masa Depan) Mulai Desember 2024')
          ax.set_xlabel('Tanggal')
          ax.set_ylabel('Harga')
          ax.legend()
          ax.grid()
  
          # Batas sumbu x
          ax.set_xlim(pd.to_datetime('2024-12-01'), pd.to_datetime('2025-01-07'))
  
          st.pyplot(fig)

