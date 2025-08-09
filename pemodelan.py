# pemodelan.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_goldfeldquandt
from scipy.stats import kstest
import warnings
warnings.filterwarnings("ignore")

# ========================
# 1. Load & Prepare Data
# ========================
def load_data(file):
    """Load CSV file, parse date, set index"""
    df = pd.read_csv(file)
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df = df.sort_values('Tanggal')
    df.set_index('Tanggal', inplace=True)
    return df

def prepare_data(df, exog_columns=None):
    """Pisahkan y dan X sesuai pilihan user"""
    y = df['Harga']
    X = df[exog_columns] if exog_columns and len(exog_columns) > 0 else None
    return y, X

# ========================
# 2. Analisis Stasioneritas
# ========================
def adf_test(series):
    """Uji Augmented Dickey-Fuller"""
    result = adfuller(series)
    return {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Stationary': result[1] < 0.05
    }

def plot_acf_pacf(series, lags=20):
    """Plot ACF dan PACF"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(series, lags=lags, ax=axes[0])
    plot_pacf(series, lags=lags, ax=axes[1])
    plt.tight_layout()
    return fig

# ========================
# 3. Pemodelan
# ========================
def fit_model(y, X=None, order=(1,0,0)):
    """Fit ARIMA atau ARIMAX sesuai eksogen"""
    if X is not None:
        model = SARIMAX(y, exog=X, order=order)
    else:
        model = ARIMA(y, order=order)
    result = model.fit()
    return result

def select_best_model(y, X=None, p_range=range(0,4), d_range=range(0,3), q_range=range(0,4)):
    """Grid search untuk AIC terkecil"""
    best_aic = np.inf
    best_order = None
    best_model = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = SARIMAX(y, exog=X, order=(p,d,q)) if X is not None else ARIMA(y, order=(p,d,q))
                    result = model.fit(disp=False)
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p,d,q)
                        best_model = result
                except:
                    continue
    return best_order, best_model

# ========================
# 4. Diagnostik Model
# ========================
def diagnostic_tests(residuals):
    """Uji KS, Ljung-Box, Goldfeld-Quandt"""
    ks_stat, ks_p = kstest(residuals, 'norm')
    lb_stat, lb_p = acorr_ljungbox(residuals, lags=[10], return_df=False)
    gq_stat, gq_p, _ = het_goldfeldquandt(residuals, np.arange(len(residuals))%2)
    
    return {
        'Kolmogorov-Smirnov': {'stat': ks_stat, 'pvalue': ks_p, 'Pass': ks_p > 0.05},
        'Ljung-Box': {'stat': lb_stat[0], 'pvalue': lb_p[0], 'Pass': lb_p[0] > 0.05},
        'Goldfeld-Quandt': {'stat': gq_stat, 'pvalue': gq_p, 'Pass': gq_p > 0.05}
    }

# ========================
# 5. Prediksi
# ========================
def forecast(model, steps=10, exog_future=None):
    """Forecast ke depan"""
    if exog_future is not None:
        pred = model.get_forecast(steps=steps, exog=exog_future)
    else:
        pred = model.get_forecast(steps=steps)
    forecast_df = pred.summary_frame()
    return forecast_df
