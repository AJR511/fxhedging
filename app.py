import streamlit as st
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Yarda FX Hedge Simulator", layout="wide")

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
# Helpers
# ----------------------------
def is_business_day(date_value) -> bool:
    ts = pd.Timestamp(date_value)
    return ts.weekday() < 5


def next_business_day(date_value):
    ts = pd.Timestamp(date_value).normalize()
    while ts.weekday() >= 5:
        ts += pd.Timedelta(days=1)
    return ts.date()


def format_notional(value):
    return f"{value:,.0f}"


def get_hedge_action(transaction_direction):
    return "Buy" if transaction_direction == "pay" else "Sell"


def get_pair_label(foreign_currency, local_currency):
    return f"{foreign_currency}/{local_currency}"


def get_pair_history(base_df, foreign_currency, local_currency):
    """
    base_df comes from USD/MXN OANDA history.
    We support only:
    - USD/MXN directly
    - MXN/USD via inversion
    """
    if foreign_currency == local_currency:
        return pd.DataFrame()

    pair_df = base_df.copy()

    if foreign_currency == "USD" and local_currency == "MXN":
        pair_df["spot"] = pair_df["spot"]
        pair_df["pair"] = "USD/MXN"
        return pair_df

    if foreign_currency == "MXN" and local_currency == "USD":
        pair_df["spot"] = 1 / pair_df["spot"]
        pair_df["pair"] = "MXN/USD"
        return pair_df

    return pd.DataFrame()


def unhedged_cash_impact(row, notional):
    return notional * row["simulated_spot"]


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
def fetch_usd_mxn_spot_data():
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
        oanda_spot_df["pair"] = "USD/MXN"
        return oanda_spot_df

    return pd.DataFrame()


# ----------------------------
# Session state
# ----------------------------
if "run_simulation" not in st.session_state:
    st.session_state.run_simulation = False

# ----------------------------
# App header
# ----------------------------
st.title("Yarda FX Hedge Simulator")
st.caption(
    "Illustrative tool only. This simulator uses historical FX moves to show how a hedge might behave."
)

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Tell us about your exposure")

today = pd.Timestamp.now().normalize()
default_settlement = next_business_day(today + pd.Timedelta(days=1))

with st.sidebar.form("hedge_form"):
    transaction_direction = st.radio(
        'My business will...',
        ("pay", "receive"),
        index=0,
        horizontal=True
    )

    foreign_currency = st.selectbox(
        "Foreign currency",
        options=["USD", "MXN"],
        index=0
    )

    settlement_date_input = st.date_input(
        "Date",
        value=default_settlement,
        min_value=default_settlement
    )

    local_currency = st.selectbox(
        "Local currency",
        options=["USD", "MXN"],
        index=1
    )

    st.markdown(
        f'My business will **{transaction_direction}** **{foreign_currency}** on '
        f'**{pd.Timestamp(settlement_date_input).strftime("%Y-%m-%d")}**, '
        f'and my business operates in **{local_currency}**.'
    )

    notional = st.number_input(
        f"Notional amount ({foreign_currency})",
        min_value=1000.0,
        value=1_000_000.0,
        step=1000.0
    )

    default_forward = 18.4568 if (foreign_currency == "USD" and local_currency == "MXN") else 0.0542

    forward_rate_input = st.number_input(
        f"Forward rate ({foreign_currency}/{local_currency})",
        min_value=0.000001,
        value=float(default_forward),
        step=0.0001,
        format="%.4f"
    )

    hedge_ratios_selection = st.multiselect(
        "Hedge ratios to compare",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        default=[0.0, 0.5, 1.0]
    )

    submitted = st.form_submit_button("What will a hedge look like?")

if submitted:
    st.session_state.run_simulation = True

if not st.session_state.run_simulation:
    st.info("Fill in the exposure details on the left and click “What will a hedge look like?”")
    st.stop()

# ----------------------------
# Validation
# ----------------------------
if foreign_currency == local_currency:
    st.warning("Foreign currency and local currency must be different.")
    st.stop()

if not is_business_day(settlement_date_input):
    st.warning("Please choose a weekday settlement date.")
    st.stop()

if not hedge_ratios_selection:
    st.warning("Please select at least one hedge ratio.")
    st.stop()

pair_label = get_pair_label(foreign_currency, local_currency)
hedge_action = get_hedge_action(transaction_direction)
settlement_dt = pd.to_datetime(settlement_date_input).normalize()
local_currency_label = local_currency

# ----------------------------
# Load spot data
# ----------------------------
with st.spinner("Fetching historical FX data from OANDA..."):
    usd_mxn_df = fetch_usd_mxn_spot_data()

if usd_mxn_df.empty:
    st.error("No data fetched from OANDA for the specified period.")
    st.stop()

spot_df = get_pair_history(usd_mxn_df, foreign_currency, local_currency)

if spot_df.empty:
    st.error("This version currently supports only USD/MXN and MXN/USD.")
    st.stop()

# ----------------------------
# Simulation logic
# ----------------------------
calendar_days = (settlement_dt - today).days

if calendar_days <= 0:
    st.warning("Settlement date must be in the future.")
    st.stop()

TRADING_DAYS_PER_YEAR = 252
tenor_trading_days = max(1, int(calendar_days * (TRADING_DAYS_PER_YEAR / 365)))
forward_rate = float(forward_rate_input)

if tenor_trading_days >= len(spot_df):
    st.error(
        f"Not enough historical data ({len(spot_df)} records) to simulate "
        f"{tenor_trading_days} trading days."
    )
    st.stop()

# ----------------------------
# Hedge recommendation
# ----------------------------
st.subheader("Recommended full hedge")

st.markdown(
    f"To fully hedge your position, your company needs to **{hedge_action}** "
    f"**{format_notional(notional)} {foreign_currency}** via a "
    f"**{pair_label} forward** maturing on **{settlement_dt.strftime('%Y-%m-%d')}**."
)

st.markdown(
    f"Based on historical data of **{pair_label}**, this is how a hedge should behave."
)

# ----------------------------
# Simulation details
# ----------------------------
st.subheader("Simulation Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Today", today.strftime("%Y-%m-%d"))

with col2:
    st.metric("Settlement Date", settlement_dt.strftime("%Y-%m-%d"))

with col3:
    st.metric("Calendar Days to Maturity", int(calendar_days))

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Effective Trading Days", int(tenor_trading_days))

with col5:
    st.metric("Notional", f"{format_notional(notional)} {foreign_currency}")

with col6:
    st.metric(f"Forward Rate ({pair_label})", f"{forward_rate:.4f}")

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
results_df["unhedged_local"] = results_df.apply(
    lambda row: unhedged_cash_impact(row, notional),
    axis=1,
)

for ratio in hedge_ratios_selection:
    results_df[f"hedged_{int(ratio * 100)}"] = results_df.apply(
        lambda row: hedged_cash_impact(row, ratio, notional, forward_rate),
        axis=1,
    )

# ----------------------------
# PnL calculation in local currency
# ----------------------------
baseline_cost = notional * forward_rate

for ratio in hedge_ratios_selection:
    if ratio == 0.0:
        value_col = "unhedged_local"
        pnl_col = "pnl_unhedged"
    else:
        value_col = f"hedged_{int(ratio * 100)}"
        pnl_col = f"pnl_{int(ratio * 100)}"

    if transaction_direction == "pay":
        results_df[pnl_col] = baseline_cost - results_df[value_col]
    else:
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
            "Worst Case PnL": f"{{:,.0f}} {local_currency_label}",
            "Best Case PnL": f"{{:,.0f}} {local_currency_label}",
            "Average PnL": f"{{:,.0f}} {local_currency_label}",
        }
    )
)

# ----------------------------
# Chart
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

ax.set_title(f"Historical Hedge Outcome Distribution: {pair_label}")
ax.set_xlabel("Hedging Strategy")
ax.set_ylabel(f"PnL ({local_currency_label})")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.7)

st.pyplot(fig)

# ----------------------------
# Optional raw data preview
# ----------------------------
with st.expander("Show raw simulation data"):
    st.dataframe(results_df)

# ----------------------------
# Optional CSV export
# ----------------------------
st.download_button(
    label="Download simulation results (CSV)",
    data=results_df.to_csv(index=False),
    file_name="fx_simulation_results.csv",
    mime="text/csv"
)
