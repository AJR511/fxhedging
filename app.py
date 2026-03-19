import streamlit as st
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="FX Hedging Simulation", layout="wide")

# ----------------------------
# OANDA API key from Streamlit secrets
# ----------------------------
try:
    OANDA_API_KEY = st.secrets["OANDA_API_KEY"]
except Exception:
    st.error("OANDA_API_KEY not found in Streamlit secrets.")
    st.stop()

client = oandapyV20.API(access_token=OANDA_API_KEY)

# ----------------------------
# Helper functions
# ----------------------------
def unhedged_cash_impact(row, notional, direction):
    future_spot = row["simulated_spot"]
    if direction == "pay":
        return notional * future_spot
    else:  # receive
        return notional * future_spot


def hedged_cash_impact(row, hedge_ratio, notional, forward_rate):
    future_spot = row["simulated_spot"]
    hedged_part = hedge_ratio * notional * forward_rate
    unhedged_part = (1 - hedge_ratio) * notional * future_spot
    return hedged_part + unhedged_part


def summarize_pnl(series):
    return {
        "best_case": round(series.max(), 0),
        "worst_case": round(series.min(), 0),
        "average": round(series.mean(), 0),
    }


# ----------------------------
# Data fetch
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_oanda_spot_data():
    instrument = "USD_MXN"
    all_oanda_data = []
    end_date = pd.Timestamp.now(tz="UTC").floor("D")
    start_date = end_date - pd.DateOffset(years=20)
    current_chunk_end = end_date

    while current_chunk_end > start_date:
        params = {
            "granularity": "D",
            "to": current_chunk_end.isoformat(),
            "count": 5000,
        }

        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(r)

        if r.response and "candles" in r.response:
            chunk_data = []

            for candle in r.response["candles"]:
                if candle.get("complete", True) and "mid" in candle:
                    chunk_data.append(
                        {
                            "date": pd.to_datetime(candle["time"]),
                            "spot": float(candle["mid"]["c"]),
                        }
                    )

            if chunk_data:
                all_oanda_data.extend(chunk_data)

                oldest_candle_time = pd.to_datetime(r.response["candles"][0]["time"])
                next_chunk_end = oldest_candle_time - pd.Timedelta(days=1)

                if next_chunk_end >= current_chunk_end:
                    break

                current_chunk_end = next_chunk_end
            else:
                break
        else:
            break

    if all_oanda_data:
        oanda_spot_df = pd.DataFrame(all_oanda_data)
        oanda_spot_df = (
            oanda_spot_df.sort_values("date")
            .drop_duplicates(subset=["date"])
            .reset_index(drop=True)
        )
        oanda_spot_df["pair"] = instrument.replace("_", "/")
        return oanda_spot_df

    return pd.DataFrame()


# ----------------------------
# App header
# ----------------------------
st.title("FX Hedging Simulation")
st.sidebar.header("Simulation Parameters")

# ----------------------------
# Inputs
# ----------------------------
direction = st.sidebar.radio("Transaction Direction", ("pay", "receive"), index=0)

notional = st.sidebar.number_input(
    "Notional Amount (USD)",
    min_value=1000,
    value=1_000_000,
    step=1000,
)

settlement_date_input = st.sidebar.date_input(
    "Settlement Date",
    value=pd.to_datetime("2026-06-30"),
)

forward_rate_input = st.sidebar.number_input(
    "Forward Rate (USD/MXN)",
    min_value=1.0,
    value=18.4568,
    step=0.0001,
    format="%.4f",
)

hedge_ratios_selection = st.sidebar.multiselect(
    "Select Hedge Ratios",
    options=[0.0, 0.25, 0.5, 0.75, 1.0],
    default=[0.0, 0.5, 1.0],
)

if not hedge_ratios_selection:
    st.sidebar.warning("Please select at least one hedge ratio.")
    st.stop()

home_currency = "MXN"

# ----------------------------
# Load spot data
# ----------------------------
with st.spinner("Fetching historical USD/MXN data from OANDA..."):
    spot_df = fetch_oanda_spot_data()

if spot_df.empty:
    st.error("No data fetched from OANDA for the specified period.")
    st.stop()

# ----------------------------
# Simulation logic
# ----------------------------
today = pd.Timestamp.now().normalize()
settlement_dt = pd.to_datetime(settlement_date_input).normalize()

calendar_days = (settlement_dt - today).days

if calendar_days <= 0:
    st.warning("Settlement date must be in the future.")
    st.stop()

TRADING_DAYS_PER_YEAR = 252
tenor_trading_days = max(1, int(calendar_days * (TRADING_DAYS_PER_YEAR / 365)))

forward_rate = forward_rate_input

# ----------------------------
# Simulation Details
# ----------------------------
st.subheader("Simulation Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Today", today.strftime("%Y-%m-%d"))

with col2:
    st.metric("Settlement Date", settlement_dt.strftime("%Y-%m-%d"))

with col3:
    st.metric("Calendar Days to Maturity", int(calendar_days))

col4, col5 = st.columns(2)

with col4:
    st.metric("Effective Trading Days for Simulation", int(tenor_trading_days))

with col5:
    st.metric(f"Forward Rate (USD/{home_currency})", f"{forward_rate:.4f}")

if tenor_trading_days >= len(spot_df):
    st.error(
        f"Not enough historical data ({len(spot_df)} records) to simulate "
        f"{tenor_trading_days} trading days."
    )
    st.stop()

# ----------------------------
# Generate scenarios
# ----------------------------
results = []

for i in range(len(spot_df) - tenor_trading_days):
    historical_start_date = spot_df.iloc[i]["date"]
    historical_end_date = spot_df.iloc[i + tenor_trading_days]["date"]
    historical_start_spot = spot_df.iloc[i]["spot"]
    historical_end_spot = spot_df.iloc[i + tenor_trading_days]["spot"]

    pct_move = (historical_end_spot / historical_start_spot) - 1

    results.append(
        {
            "start_date": historical_start_date,
            "end_date_historical": historical_end_date,
            "historical_start_spot": historical_start_spot,
            "historical_end_spot": historical_end_spot,
            "pct_move": pct_move,
        }
    )

results_df = pd.DataFrame(results)

if results_df.empty:
    st.error("No simulation scenarios could be generated.")
    st.stop()

mean_move = results_df["pct_move"].mean()
results_df["demeaned_move"] = results_df["pct_move"] - mean_move
results_df["simulated_spot"] = forward_rate * (1 + results_df["demeaned_move"])

# ----------------------------
# Cash impact
# ----------------------------
results_df["unhedged_mxn"] = results_df.apply(
    lambda row: unhedged_cash_impact(row, notional, direction),
    axis=1,
)

for ratio in hedge_ratios_selection:
    results_df[f"hedged_{int(ratio * 100)}"] = results_df.apply(
        lambda row: hedged_cash_impact(row, ratio, notional, forward_rate),
        axis=1,
    )

# ----------------------------
# PnL calculation
# ----------------------------
baseline_cost = notional * forward_rate

for ratio in hedge_ratios_selection:
    if ratio == 0.0:
        value_col = "unhedged_mxn"
        pnl_col = "pnl_unhedged"
    else:
        value_col = f"hedged_{int(ratio * 100)}"
        pnl_col = f"pnl_{int(ratio * 100)}"

    if direction == "pay":
        results_df[pnl_col] = baseline_cost - results_df[value_col]
    else:  # receive
        results_df[pnl_col] = results_df[value_col] - baseline_cost

# ----------------------------
# Summary table
# ----------------------------
st.header("Simulation Summary")

summary_data = []

for ratio in hedge_ratios_selection:
    pnl_col_name = f"pnl_{int(ratio * 100)}" if ratio > 0 else "pnl_unhedged"
    summary = summarize_pnl(results_df[pnl_col_name])

    summary_data.append(
        {
            "Strategy": f"{int(ratio * 100)}% Hedged" if ratio > 0 else "Unhedged",
            "Worst Case PnL": summary["worst_case"],
            "Best Case PnL": summary["best_case"],
            "Average PnL": summary["average"],
        }
    )

summary_df = pd.DataFrame(summary_data)

st.dataframe(
    summary_df.set_index("Strategy").style.format(
        {
            "Worst Case PnL": "{:,.0f} MXN",
            "Best Case PnL": "{:,.0f} MXN",
            "Average PnL": "{:,.0f} MXN",
        }
    )
)

# ----------------------------
# Violin plot
# ----------------------------
st.header("PnL Distribution Across Hedging Strategies")

pnl_combined_list = []

for ratio in hedge_ratios_selection:
    pnl_col_name = f"pnl_{int(ratio * 100)}" if ratio > 0 else "pnl_unhedged"
    strategy_label = f"{int(ratio * 100)}% Hedged" if ratio > 0 else "Unhedged"

    pnl_combined_list.append(
        pd.DataFrame(
            {
                "PnL": results_df[pnl_col_name],
                "Strategy": strategy_label,
            }
        )
    )

pnl_combined = pd.concat(pnl_combined_list, ignore_index=True)

fig, ax = plt.subplots(figsize=(12, 7))
sns.violinplot(
    x="Strategy",
    y="PnL",
    data=pnl_combined,
    inner="quartile",
    ax=ax,
    hue="Strategy",
    legend=False,
)

ax.set_title("PnL Distribution Across Hedging Strategies")
ax.set_xlabel("Hedging Strategy")
ax.set_ylabel(f"PnL ({home_currency})")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.7)

st.pyplot(fig)

# ----------------------------
# Optional raw data preview
# ----------------------------
with st.expander("Show raw simulation data"):
    st.dataframe(results_df)
