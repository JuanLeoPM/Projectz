import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import coint

st.set_page_config(page_title="Finance tools", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #000000;
        color: #f5f5f5;
    }
    [data-testid="stAppViewContainer"] {
        background: #000000;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    [data-testid="stSidebar"] {
        background: #050505;
        border-right: 1px solid #1f1f1f;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, h4, h5, h6, p, div, label, span {
        color: #f5f5f5 !important;
    }
    .stMarkdown, .stText, .stCaption {
        color: #d9d9d9 !important;
    }
    [data-testid="stMetric"] {
        background: #0d0d0d;
        border: 1px solid #202020;
        border-radius: 14px;
        padding: 0.75rem;
    }
    [data-testid="stDataFrame"], .stTable {
        background: #0a0a0a;
        border-radius: 12px;
    }
    .stAlert {
        background: #101010;
        color: #f5f5f5;
        border: 1px solid #242424;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111111;
        border-radius: 10px 10px 0 0;
        color: #f5f5f5;
    }
    .stButton>button, .stDownloadButton>button {
        background: #111111;
        color: white;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
    }
    .stTextInput>div>div>input, .stNumberInput input, .stDateInput input {
        background-color: #0d0d0d !important;
        color: #f5f5f5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Finance tools")
st.caption("Juan Leonardo Patiño Martinez")

# -----------------------------
# Utilities
# -----------------------------

def download_prices(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            data = data.xs(data.columns.levels[0][0], axis=1, level=0)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=tickers[0] if isinstance(tickers, list) else str(tickers))
    return data.dropna(how="all")


def annualized_return(returns):
    return returns.mean() * 252


def annualized_vol(returns):
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns, rf=0.0):
    vol = annualized_vol(returns)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (annualized_return(returns) - rf) / vol


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def simulate_gbm_paths(S0, mu, sigma, T, steps, n_sims, seed=42):
    np.random.seed(seed)
    dt = T / steps
    z = np.random.normal(size=(steps, n_sims))
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.vstack([np.zeros(n_sims), increments]).cumsum(axis=0)
    return S0 * np.exp(log_paths)


def portfolio_stats(weights, mean_returns, cov_matrix, rf=0.0):
    port_return = np.dot(weights, mean_returns) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else np.nan
    return port_return, port_vol, sharpe


def maximize_sharpe(returns, rf=0.0):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    n = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init = np.array([1 / n] * n)

    def neg_sharpe(weights):
        _, _, s = portfolio_stats(weights, mean_returns, cov_matrix, rf)
        return -s

    result = minimize(neg_sharpe, init, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x, mean_returns, cov_matrix


def efficient_frontier_points(mean_returns, cov_matrix, points=30):
    n = len(mean_returns)
    target_returns = np.linspace(mean_returns.min() * 252, mean_returns.max() * 252, points)
    frontier = []
    bounds = tuple((0, 1) for _ in range(n))

    for tr in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x, tr=tr: np.dot(x, mean_returns) * 252 - tr},
        )
        init = np.array([1 / n] * n)

        def min_vol(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

        res = minimize(min_vol, init, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success:
            vol = min_vol(res.x)
            frontier.append((tr, vol))

    return pd.DataFrame(frontier, columns=["Return", "Volatility"])


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Configuration")
start_date = st.sidebar.date_input("Start date", value=date.today() - timedelta(days=365 * 3))
end_date = st.sidebar.date_input("End date", value=date.today())
rf_rate = st.sidebar.number_input("Risk-free rate (annual, decimal)", value=0.04, step=0.005, format="%.3f")

app_mode = st.sidebar.radio(
    "Choose module",
    [
        "Overview",
        "1. Price Prediction",
        "2. Portfolio Optimization",
        "3. Monte Carlo Options",
        "4. Pairs Trading",
        "5. Value at Risk",
    ],
)

# -----------------------------
# Overview
# -----------------------------

if app_mode == "Overview":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("What this app improves")
        st.markdown(
            """
Instead of 5 disconnected notebook exercises, this app upgrades them into a portfolio project:

- **Unified interface** with one data pipeline.
- **Interactive charts** so it feels like a real product.
- **Better metrics** than the original weekend ideas.
- **Reusable finance functions** for scaling later.
- **Portfolio-ready storytelling** so recruiters can understand what you built.
            """
        )
        st.subheader("Best way to present it")
        st.markdown(
            """
Call this a **Quant Research Sandbox** or **Weekend Quant Lab**.

Show it as:
1. a Streamlit web app,
2. a GitHub repo with clean README,
3. short writeups for each module,
4. screenshots or a demo video.
            """
        )
    with col2:
        roadmap = pd.DataFrame(
            {
                "Module": [
                    "Price Prediction",
                    "Portfolio Optimization",
                    "Monte Carlo",
                    "Pairs Trading",
                    "VaR",
                ],
                "What it teaches": [
                    "Features, regression, forecasting",
                    "Risk/return tradeoff",
                    "Stochastic simulation",
                    "Stat arb and backtesting logic",
                    "Risk measurement",
                ],
            }
        )
        st.dataframe(roadmap, use_container_width=True)

    st.subheader("How to improve it even more")
    st.markdown(
        """
- Add **walk-forward validation** to the prediction model.
- Add **constraints** and **transaction costs** to optimization.
- Add **Greeks** and **variance reduction** to the option pricer.
- Add **real PnL, costs, and stop-loss rules** to pairs trading.
- Add **Expected Shortfall (CVaR)** to the VaR module.
        """
    )

# -----------------------------
# 1. Price Prediction
# -----------------------------

elif app_mode == "1. Price Prediction":
    st.header("1. Stock Price Prediction with Linear Regression")
    ticker = st.text_input("Ticker", value="AAPL").upper().strip()
    horizon = st.slider("Prediction horizon (days ahead)", min_value=1, max_value=10, value=1)

    px_data = download_prices([ticker], start_date, end_date)
    if px_data.empty:
        st.error("No price data found.")
    else:
        df = px_data.rename(columns={px_data.columns[0]: "close"}).copy()
        df["ret_1d"] = df["close"].pct_change()
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_10"] = df["close"].rolling(10).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["vol_10"] = df["ret_1d"].rolling(10).std()
        df["mom_5"] = df["close"] / df["close"].shift(5) - 1
        df["target"] = df["close"].shift(-horizon)
        model_df = df.dropna().copy()

        X = model_df[["close", "ma_5", "ma_10", "ma_20", "vol_10", "mom_5"]]
        y = model_df["target"]

        split_idx = int(len(model_df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        latest_features = X.iloc[[-1]]
        next_pred = model.predict(latest_features)[0]
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted future close", f"{next_pred:,.2f}")
        c2.metric("MAE", f"{mae:,.3f}")
        c3.metric("R²", f"{r2:,.3f}")

        plot_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": preds,
        }, index=y_test.index)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Actual"], name="Actual"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Predicted"], name="Predicted"))
        fig.update_layout(height=500, title=f"{ticker}: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

        coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
        st.subheader("Feature importance proxy")
        st.dataframe(coef_df.sort_values("Coefficient", ascending=False), use_container_width=True)

        st.info(
            "Upgrade path: replace linear regression with XGBoost, LightGBM, TFT, or a regime-aware model with walk-forward testing."
        )

# -----------------------------
# 2. Portfolio Optimization
# -----------------------------

elif app_mode == "2. Portfolio Optimization":
    st.header("2. Portfolio Optimization Tool")
    tickers_text = st.text_input("Tickers (comma separated)", value="AAPL,MSFT,NVDA,GOOGL,AMZN")
    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]

    prices = download_prices(tickers, start_date, end_date)
    if prices.empty or prices.shape[1] < 2:
        st.error("Need at least two valid tickers.")
    else:
        returns = prices.pct_change().dropna()
        weights, mean_returns, cov_matrix = maximize_sharpe(returns, rf=rf_rate)
        port_return, port_vol, port_sharpe = portfolio_stats(weights, mean_returns, cov_matrix, rf=rf_rate)
        frontier = efficient_frontier_points(mean_returns, cov_matrix, points=40)

        c1, c2, c3 = st.columns(3)
        c1.metric("Expected annual return", f"{port_return:.2%}")
        c2.metric("Expected annual volatility", f"{port_vol:.2%}")
        c3.metric("Sharpe ratio", f"{port_sharpe:.2f}")

        weights_df = pd.DataFrame({"Ticker": returns.columns, "Weight": weights}).sort_values("Weight", ascending=False)
        st.subheader("Optimal weights")
        st.dataframe(weights_df.style.format({"Weight": "{:.2%}"}), use_container_width=True)

        fig = px.scatter(frontier, x="Volatility", y="Return", title="Efficient Frontier")
        fig.add_trace(go.Scatter(x=[port_vol], y=[port_return], mode="markers", name="Max Sharpe Portfolio", marker_size=14))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        cum_returns = (1 + returns @ weights).cumprod()
        fig2 = px.line(cum_returns, title="Optimized Portfolio Growth of $1")
        fig2.update_layout(height=450, xaxis_title="Date", yaxis_title="Portfolio Value")
        st.plotly_chart(fig2, use_container_width=True)

        st.info(
            "Upgrade path: add rebalancing, max position limits, sector caps, transaction costs, and Black-Litterman views."
        )

# -----------------------------
# 3. Monte Carlo Options
# -----------------------------

elif app_mode == "3. Monte Carlo Options":
    st.header("3. Monte Carlo Simulation for Options Pricing")
    ticker = st.text_input("Underlying ticker", value="SPY").upper().strip()
    option_type = st.selectbox("Option type", ["call", "put"])
    strike_pct = st.slider("Strike as % of current spot", min_value=70, max_value=130, value=100)
    maturity_days = st.slider("Maturity (days)", min_value=7, max_value=365, value=90)
    n_sims = st.slider("Number of simulations", min_value=1000, max_value=50000, step=1000, value=10000)

    prices = download_prices([ticker], start_date, end_date)
    if prices.empty:
        st.error("No data found.")
    else:
        s = float(prices.iloc[-1, 0])
        rets = prices.iloc[:, 0].pct_change().dropna()
        sigma = rets.std() * np.sqrt(252)
        mu = rets.mean() * 252
        T = maturity_days / 365
        K = s * strike_pct / 100
        steps = max(20, maturity_days)

        paths = simulate_gbm_paths(s, mu, sigma, T, steps, n_sims)
        terminal = paths[-1]

        if option_type == "call":
            payoffs = np.maximum(terminal - K, 0)
            bs_price = black_scholes_call(s, K, T, rf_rate, sigma)
        else:
            payoffs = np.maximum(K - terminal, 0)
            bs_price = black_scholes_put(s, K, T, rf_rate, sigma)

        mc_price = np.exp(-rf_rate * T) * payoffs.mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spot", f"{s:,.2f}")
        c2.metric("Strike", f"{K:,.2f}")
        c3.metric("Monte Carlo Price", f"{mc_price:,.4f}")
        c4.metric("Black-Scholes", f"{bs_price:,.4f}")

        sample_paths = pd.DataFrame(paths[:, :100])
        fig = px.line(sample_paths, title="Sample Simulated Price Paths")
        fig.update_layout(height=500, xaxis_title="Step", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        hist_fig = px.histogram(terminal, nbins=60, title="Distribution of Terminal Prices")
        hist_fig.update_layout(height=450)
        st.plotly_chart(hist_fig, use_container_width=True)

        st.info(
            "Upgrade path: add antithetic variates, control variates, Greeks, local volatility, stochastic volatility, and American option approximations."
        )

# -----------------------------
# 4. Pairs Trading
# -----------------------------

elif app_mode == "4. Pairs Trading":
    st.header("4. Pairs Trading Strategy Backtest")
    c1, c2 = st.columns(2)
    ticker_a = c1.text_input("Ticker A", value="KO").upper().strip()
    ticker_b = c2.text_input("Ticker B", value="PEP").upper().strip()
    entry_z = st.slider("Entry z-score", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
    exit_z = st.slider("Exit z-score", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

    prices = download_prices([ticker_a, ticker_b], start_date, end_date).dropna()
    if prices.empty or prices.shape[1] < 2:
        st.error("Need two valid tickers with overlapping history.")
    else:
        pval = coint(prices.iloc[:, 0], prices.iloc[:, 1])[1]

        y = prices.iloc[:, 0]
        x = prices.iloc[:, 1]
        beta = np.polyfit(x, y, 1)[0]
        spread = y - beta * x
        zscore = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()

        signals = pd.DataFrame(index=prices.index)
        signals["zscore"] = zscore
        signals["position"] = 0
        signals.loc[zscore > entry_z, "position"] = -1
        signals.loc[zscore < -entry_z, "position"] = 1
        signals.loc[zscore.abs() < exit_z, "position"] = 0
        signals["position"] = signals["position"].replace(to_replace=0, method="ffill").fillna(0)

        pair_ret = y.pct_change().fillna(0) - beta * x.pct_change().fillna(0)
        strategy_ret = signals["position"].shift(1).fillna(0) * (-pair_ret)
        equity = (1 + strategy_ret).cumprod()

        c1, c2, c3 = st.columns(3)
        c1.metric("Cointegration p-value", f"{pval:.4f}")
        c2.metric("Strategy total return", f"{equity.iloc[-1] - 1:.2%}")
        c3.metric("Strategy Sharpe", f"{sharpe_ratio(strategy_ret, rf=0):.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=signals.index, y=signals["zscore"], name="Z-score"))
        fig.add_hline(y=entry_z)
        fig.add_hline(y=-entry_z)
        fig.add_hline(y=exit_z, line_dash="dot")
        fig.add_hline(y=-exit_z, line_dash="dot")
        fig.update_layout(height=450, title="Spread Z-score")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(equity, title="Pairs Strategy Equity Curve")
        fig2.update_layout(height=450, yaxis_title="Equity")
        st.plotly_chart(fig2, use_container_width=True)

        st.info(
            "Upgrade path: hedge ratio with rolling OLS/Kalman filter, proper trade execution, slippage, and a market-neutral portfolio construction layer."
        )

# -----------------------------
# 5. Value at Risk
# -----------------------------

elif app_mode == "5. Value at Risk":
    st.header("5. Value at Risk (VaR) Calculator")
    tickers_text = st.text_input("Portfolio tickers", value="AAPL,MSFT,NVDA")
    weights_text = st.text_input("Weights (comma separated, same order)", value="0.3,0.4,0.3")
    confidence = st.slider("Confidence level", min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    portfolio_value = st.number_input("Portfolio value ($)", min_value=1000.0, value=100000.0, step=1000.0)

    tickers = [x.strip().upper() for x in tickers_text.split(",") if x.strip()]
    weights = np.array([float(x.strip()) for x in weights_text.split(",") if x.strip()])

    prices = download_prices(tickers, start_date, end_date)
    if prices.empty or len(tickers) != len(weights):
        st.error("Please make sure tickers and weights are valid and aligned.")
    else:
        weights = weights / weights.sum()
        returns = prices.pct_change().dropna()
        port_returns = returns @ weights

        alpha = 1 - confidence
        hist_var_pct = -np.percentile(port_returns, alpha * 100)
        mu = port_returns.mean()
        sigma = port_returns.std()
        param_var_pct = -(mu + sigma * norm.ppf(alpha))
        cvar_pct = -port_returns[port_returns <= np.percentile(port_returns, alpha * 100)].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Historical VaR ({int(confidence*100)}%)", f"${hist_var_pct * portfolio_value:,.2f}")
        c2.metric("Parametric VaR", f"${param_var_pct * portfolio_value:,.2f}")
        c3.metric("CVaR / Expected Shortfall", f"${cvar_pct * portfolio_value:,.2f}")

        fig = px.histogram(port_returns, nbins=60, title="Portfolio Daily Return Distribution")
        fig.add_vline(x=-hist_var_pct, line_dash="dash", annotation_text="Historical VaR")
        fig.add_vline(x=-param_var_pct, line_dash="dot", annotation_text="Parametric VaR")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        dd = (1 + port_returns).cumprod()
        dd = dd / dd.cummax() - 1
        st.metric("Max Drawdown", f"{dd.min():.2%}")

        st.info(
            "Upgrade path: add Monte Carlo VaR, stress scenarios, factor shocks, liquidity haircuts, and horizon scaling."
        )



