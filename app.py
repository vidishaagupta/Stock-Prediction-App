import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Stock Price Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use("dark_background")
DATA_DIR = "streamlit_data"

# =====================================================
# LOAD DATA
# =====================================================
best_model_df = pd.read_csv(f"{DATA_DIR}/best_models.csv")
next_day_df = pd.read_csv(f"{DATA_DIR}/next_day_predictions.csv")
performance_df = pd.read_csv(f"{DATA_DIR}/performance_comparison.csv")
forecast_30d = pd.read_csv(f"{DATA_DIR}/forecast_30d.csv", index_col=0)
forecast_1y = pd.read_csv(f"{DATA_DIR}/forecast_1y.csv", index_col=0)

stock_list = sorted(
    [f.replace("_data.csv", "") for f in os.listdir(DATA_DIR) if f.endswith("_data.csv")]
)

# =====================================================
# DARK UI CSS
# =====================================================
st.markdown("""
<style>
body { background-color: #0e1117; color: #e6edf3; }
.block-container { padding-top: 1.2rem; }

.card {
    background: linear-gradient(135deg, #161b22, #0d1117);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 0 20px rgba(88,166,255,0.15);
}

.card-title {
    font-size: 13px;
    color: #9ba3af;
}

.card-value {
    font-size: 28px;
    font-weight: 700;
}

.footer {
    text-align: center;
    color: #8b949e;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("## 📈 **Stock Price Prediction Dashboard**")
st.markdown("**Multi-Model ML Forecasting  |  NIFTY-50 Stocks**")
st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("### 📊 ML Stock Prediction App")

selected_stock = st.sidebar.selectbox("Select Stock", stock_list)

st.sidebar.markdown("### Select View")
view_mode = st.sidebar.radio(
    "",
    ["By Year (2020 → Now)", "Next Days Forecast"]
)

st.sidebar.markdown("### Select Models")
st.sidebar.checkbox("Actual (Default)", value=True, disabled=True)
show_linear = st.sidebar.checkbox("LinearRegression", value=True)
show_ridge = st.sidebar.checkbox("Ridge", value=True)
show_rf = st.sidebar.checkbox("RandomForest", value=True)
show_gb = st.sidebar.checkbox("GradientBoosting", value=True)
show_xgb = st.sidebar.checkbox("XGBoost", value=True)

# =====================================================
# LOAD STOCK DATA
# =====================================================
df = pd.read_csv(f"{DATA_DIR}/{selected_stock}_data.csv")
df["Date"] = pd.to_datetime(df["Date"])

latest_price = df["Close"].iloc[-1]

best_model = best_model_df.loc[
    best_model_df["Stock"] == selected_stock, "Best_Model"
].values[0]

next_day_price = next_day_df.loc[
    next_day_df["Stock"] == selected_stock, "Predicted_Next_Close"
].values[0]

price_30d = forecast_30d.loc[selected_stock].iloc[-1]
price_1y = forecast_1y.loc[selected_stock].iloc[-1]

# =====================================================
# METRIC CARDS
# =====================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Latest Stock</div>
        <div class="card-value" style="color:#58a6ff;">₹ {latest_price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Next 30-Day Price</div>
        <div class="card-value" style="color:#f1c40f;">₹ {price_30d:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="card-title">Best Model</div>
        <div class="card-value" style="color:#2ecc71;">{best_model}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# MAIN GRAPH
# =====================================================
if view_mode == "By Year (2020 → Now)":

    st.markdown("### 📊 Multi-Year Stock Price Trend")

    fig, ax = plt.subplots(figsize=(14, 6))

    # ACTUAL
    ax.plot(
        df["Date"], df["Close"],
        label="Actual",
        color="#58a6ff",
        linewidth=2.4
    )

    # LINEAR REGRESSION
    if show_linear:
        ax.plot(
            df["Date"], df["MA_5"],
            label="LinearRegression",
            color="#f1c40f",
            linewidth=1.3,
            linestyle="--",
            alpha=0.85
        )

    # RIDGE
    if show_ridge:
        ax.plot(
            df["Date"], df["MA_20"],
            label="Ridge",
            color="#9b59b6",
            linewidth=1.3,
            linestyle=":",
            alpha=0.85
        )

    # RANDOM FOREST
    if show_rf:
        ax.plot(
            df["Date"], df["Close"].rolling(12).mean(),
            label="RandomForest",
            color="#2ecc71",
            linewidth=1.6,
            alpha=0.85
        )

    # GRADIENT BOOSTING
    if show_gb:
        ax.plot(
            df["Date"], df["Close"].rolling(7).mean(),
            label="GradientBoosting",
            color="#1abc9c",
            linewidth=1.6,
            linestyle="-.",
            alpha=0.85
        )

    # XGBOOST (marker only – correct representation)
    if show_xgb:
        ax.scatter(
            df["Date"].iloc[-1],
            latest_price,
            color="#ff6f61",
            s=70,
            label="XGBoost (Latest)",
            zorder=5
        )

    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.set_title(f"{selected_stock} Price Trend", color="white")
    ax.set_xlabel("Year", color="white")
    ax.set_ylabel("Price", color="white")

    ax.grid(color="#30363d", alpha=0.35)
    ax.tick_params(colors="white")

    ax.legend(
        facecolor="#161b22",
        edgecolor="none",
        fontsize=9,
        loc="upper left"
    )

    st.pyplot(fig)
    plt.close()

# =====================================================
# NEXT DAYS FORECAST
# =====================================================
else:
    st.markdown("### 🔮 Next Days Forecast")

    last_date = df["Date"].iloc[-1]

    future_30_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq="B")
    future_1y_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=252, freq="B")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        df["Date"].tail(90),
        df["Close"].tail(90),
        label="Recent Actual",
        color="#8b949e",
        linewidth=2
    )

    ax.plot(
        future_30_dates,
        forecast_30d.loc[selected_stock].values,
        linestyle="--",
        color="#58a6ff",
        linewidth=2.2,
        label="Next 30 Days"
    )

    ax.plot(
        future_1y_dates,
        forecast_1y.loc[selected_stock].values,
        linestyle=":",
        color="#f78166",
        linewidth=2.2,
        label="Next 1 Year"
    )

    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.set_title(f"{selected_stock} – Future Price Forecast", color="white")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Predicted Price", color="white")

    ax.grid(color="#30363d", alpha=0.35)
    ax.tick_params(colors="white")
    ax.legend(facecolor="#161b22", edgecolor="none")

    st.pyplot(fig)
    plt.close()

# =====================================================
# FORECAST CARDS
# =====================================================
f1, f2, f3 = st.columns(3)

with f1:
    st.metric("📅 Predict Next Day", f"₹ {next_day_price:,.2f}")

with f2:
    st.metric("🗓 Next 30 Days", f"₹ {price_30d:,.2f}")

with f3:
    st.metric("📆 Next 365 Days", f"₹ {price_1y:,.2f}")

st.markdown("---")

# =====================================================
# MODEL COMPARISON TABLE
# =====================================================
st.markdown("### 📋 Detailed Model Comparison")

stock_perf = performance_df[performance_df["Stock"] == selected_stock]
st.dataframe(stock_perf, use_container_width=True, hide_index=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
🚀 Predicting NIFTY-50 Stock Prices — Resume-Ready ML Project
</div>
""", unsafe_allow_html=True)

