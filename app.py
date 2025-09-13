# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Optional ARIMA import (may not be installed)
try:
    from statsmodels.tsa.arima.model import ARIMA
    statsmodels_available = True
except Exception:
    statsmodels_available = False

st.set_page_config(page_title="World Happiness Explorer", layout="wide")

# -----------------------
# Helpers
# -----------------------
@st.cache_data
def load_data(path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif path:
        df = pd.read_csv(path)
    else:
        st.error("No data source provided.")
        return None
    return df

def clean_basic(df):
    df = df.copy()
    # Standardize country names
    df['Country'] = df['Country'].replace(['Unknown', 'unknown', 'None', None], np.nan)
    # drop rows without Country
    df = df.dropna(subset=['Country'])
    # try convert Year to int if possible
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    # Impute numeric columns with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    # drop exact duplicates
    df = df.drop_duplicates()
    return df

def show_correlation_plot(df, numeric_only=True):
    if numeric_only:
        mat = df.select_dtypes(include=[np.number]).corr()
    else:
        mat = df.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(mat, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def run_clustering(df, n_clusters=3):
    features = ["GDP_per_Capita", "Social_Support", "Life_Expectancy"]
    existing = [c for c in features if c in df.columns]
    if len(existing) < 2:
        st.warning("Not enough numeric features for clustering. Need at least 2 of: " + ", ".join(features))
        return df, None, None
    X = df[existing].fillna(df[existing].mean())
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    df['Cluster'] = labels
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    return df, X_pca, existing

def train_regression(df):
    features = ["GDP_per_Capita","Social_Support","Life_Expectancy","Freedom","Generosity","Corruption"]
    X_cols = [c for c in features if c in df.columns]
    if len(X_cols) == 0:
        st.warning("No features available for regression.")
        return None
    X = df[X_cols].fillna(df[X_cols].mean())
    y = df['Happiness_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    coeffs = pd.Series(model.coef_, index=X_cols).sort_values(key=abs, ascending=False)
    return {'model': model, 'r2': r2, 'rmse': rmse, 'coeffs': coeffs, 'X_test':X_test, 'y_test':y_test, 'y_pred':y_pred}

def forecast_country(df, country, steps=3):
    sub = df[df['Country'] == country].sort_values('Year').copy()
    if sub.empty:
        st.error(f"No records found for {country}")
        return None
    # Ensure Year is datetime index for ARIMA
    if 'Year' in sub.columns:
        try:
            sub['Year_dt'] = pd.to_datetime(sub['Year'].astype(int).astype(str), format='%Y')
            sub.set_index('Year_dt', inplace=True)
            ts = sub['Happiness_Score'].astype(float)
        except Exception:
            ts = sub['Happiness_Score'].astype(float).reset_index(drop=True)
    else:
        ts = sub['Happiness_Score'].astype(float).reset_index(drop=True)

    if statsmodels_available:
        # ARIMA
        try:
            model = ARIMA(ts, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=steps)
            return forecast
        except Exception as e:
            st.warning("ARIMA failed: " + str(e) + "\nFalling back to linear trend forecast.")
    # fallback: linear trend on Year (if Year numeric) else index
    if 'Year' in sub.columns and pd.api.types.is_numeric_dtype(sub['Year']):
        X = sub[['Year']].values
        y = sub['Happiness_Score'].values
        lr = LinearRegression()
        lr.fit(X, y)
        last_year = int(sub['Year'].max())
        future_years = np.array([[last_year + i] for i in range(1, steps+1)])
        preds = lr.predict(future_years)
        preds_index = [last_year + i for i in range(1, steps+1)]
        return pd.Series(preds, index=preds_index)
    else:
        # use simple linear on index
        X = np.arange(len(ts)).reshape(-1,1)
        y = ts.values
        lr = LinearRegression()
        lr.fit(X, y)
        future_X = np.arange(len(ts), len(ts)+steps).reshape(-1,1)
        preds = lr.predict(future_X)
        preds_index = list(range(len(ts), len(ts)+steps))
        return pd.Series(preds, index=preds_index)

# -----------------------
# UI
# -----------------------
st.title("🌍 World Happiness Explorer")
st.write("Interactive dashboard for your World Happiness Index project. Upload a dataset or use the default cleaned CSV.")

# Sidebar: upload or load
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV (cleaned or uncleaned)", type=['csv'])
use_demo = st.sidebar.checkbox("Load demo sample (if available)", value=True)
data_path = "world_happiness_cleaned.csv" if use_demo else None

df_raw = load_data(path=data_path, uploaded_file=uploaded_file)
if df_raw is None:
    st.stop()

st.sidebar.markdown(f"Rows: {df_raw.shape[0]}  |  Columns: {df_raw.shape[1]}")

# Basic cleaning and preview
st.header("1) Data Preview & Basic Cleaning")
st.write("Raw data (first 10 rows):")
st.dataframe(df_raw.head(10))

df = clean_basic(df_raw)
st.write("After basic cleaning (first 10 rows):")
st.dataframe(df.head(10))

# Download cleaned csv
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download cleaned dataset", data=csv, file_name="world_happiness_cleaned_for_streamlit.csv")

# EDA
st.header("2) Exploratory Data Analysis (EDA)")
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Distribution of Happiness Score")
    if 'Happiness_Score' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['Happiness_Score'], bins=25, kde=True, ax=ax)
        ax.set_xlabel("Happiness Score")
        st.pyplot(fig)
    else:
        st.write("Happiness_Score column not found.")

with col2:
    st.subheader("Correlation Heatmap (numeric features)")
    show_correlation_plot(df)

# Top/Bottom countries
st.subheader("Top / Bottom countries by average Happiness Score")
if 'Country' in df.columns and 'Happiness_Score' in df.columns:
    avg_country = df.groupby('Country')['Happiness_Score'].mean().sort_values(ascending=False)
    top10 = avg_country.head(10)
    bottom10 = avg_country.tail(10)
    c1, c2 = st.columns(2)
    with c1:
        st.write("Top 10")
        st.bar_chart(top10)
    with c2:
        st.write("Bottom 10")
        st.bar_chart(bottom10)
else:
    st.write("Country / Happiness_Score columns required.")

# Choropleth
if 'Country' in df.columns and 'Happiness_Score' in df.columns:
    st.subheader("World map: Average Happiness Score by Country")
    avg_country_df = df.groupby('Country')['Happiness_Score'].mean().reset_index()
    try:
        fig = px.choropleth(avg_country_df, locations="Country", locationmode="country names",
                            color="Happiness_Score", hover_name="Country",
                            color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning("Choropleth failed (country mapping). Ensure country names are standard. " + str(e))

# Clustering
st.header("3) Clustering (MiniBatchKMeans) + PCA")
n_clusters = st.slider("Number of clusters", 2, 6, 3)
df_clustered, X_pca, cluster_features = run_clustering(df, n_clusters=n_clusters)
if X_pca is not None:
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=df_clustered['Cluster'], cmap='Set1', s=40)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("PCA projection of clusters")
    st.pyplot(fig)
    st.write("Cluster counts:")
    st.write(df_clustered['Cluster'].value_counts())
    # map labels to High/Medium/Low by average happiness
    cluster_avg = df_clustered.groupby('Cluster')['Happiness_Score'].mean().sort_values()
    mapping = {}
    labels = ["Low Happiness", "Medium Happiness", "High Happiness"]
    # create mapping from cluster id sorted by avg -> label
    for i, cluster_id in enumerate(cluster_avg.index):
        label = labels[i] if i < len(labels) else f"Cluster {i}"
        mapping[int(cluster_id)] = label
    df_clustered['Happiness_Level'] = df_clustered['Cluster'].map(mapping)
    st.dataframe(df_clustered[['Country','Year','Happiness_Score','Cluster','Happiness_Level']].head(10))

# Regression Modeling
st.header("4) Predictive Modeling (Linear Regression)")
reg_result = train_regression(df)
if reg_result:
    st.write(f"R² Score: **{reg_result['r2']:.3f}**")
    st.write(f"RMSE: **{reg_result['rmse']:.3f}**")
    st.subheader("Feature coefficients (by absolute importance)")
    st.dataframe(reg_result['coeffs'].rename("Coefficient").to_frame())

    # scatter actual vs predicted
    fig, ax = plt.subplots()
    ax.scatter(reg_result['y_test'], reg_result['y_pred'], alpha=0.6)
    ax.plot([reg_result['y_test'].min(), reg_result['y_test'].max()],
            [reg_result['y_test'].min(), reg_result['y_test'].max()], 'r--')
    ax.set_xlabel("Actual Happiness Score")
    ax.set_ylabel("Predicted Happiness Score")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)

# Time-series Forecasting
st.header("5) Time-Series Forecasting (Country-level)")
country_list = sorted(df['Country'].unique())
selected_country = st.selectbox("Select a country for forecasting", country_list)
forecast_steps = st.slider("Forecast steps (years)", 1, 5, 3)

if st.button("Run Forecast"):
    forecast = forecast_country(df, selected_country, steps=forecast_steps)
    if forecast is not None:
        st.subheader(f"Forecast for {selected_country}")
        st.write(forecast)
        # plot historic and forecast
        hist = df[df['Country'] == selected_country].sort_values('Year')
        if 'Year' in hist.columns:
            try:
                hist_plot = hist.copy()
                hist_plot['Year'] = hist_plot['Year'].astype(int)
                # plot
                fig, ax = plt.subplots()
                ax.plot(hist_plot['Year'], hist_plot['Happiness_Score'], marker='o', label='Historical')
                # forecast index may be datetime or numeric
                try:
                    idx = list(forecast.index)
                    ax.plot(idx, forecast.values, marker='x', linestyle='--', label='Forecast')
                except Exception:
                    ax.plot(range(hist_plot['Year'].max()+1, hist_plot['Year'].max()+1+len(forecast)),
                            forecast.values, marker='x', linestyle='--', label='Forecast')
                ax.set_xlabel("Year")
                ax.set_ylabel("Happiness Score")
                ax.legend()
                st.pyplot(fig)
            except Exception as e:
                st.warning("Plotting historic+forecast failed: " + str(e))

st.markdown("---")
st.write("Built with ❤️ — you can extend this app (Streamlit) to add more models, deploy to Streamlit Cloud, or export results.")
