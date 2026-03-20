import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Yarda FX Hedge Simulator", layout="wide")

# ----------------------------
# Secrets
# ----------------------------
try:
    FRED_API_KEY = st.secrets["FRED_API_KEY"]
except Exception:
    st.error("FRED_API_KEY not found in Streamlit secrets.")
    st.stop()

try:
    BANXICO_API_TOKEN = st.secrets["BANXICO_API_TOKEN"]
except Exception:
    st.error("BANXICO_API_TOKEN not found in Streamlit secrets.")
    st.stop()

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


def linear_interpolate(x, x0, y0, x1, y1):
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))


def interpolate_mxn_rate(target_days, tiie_28_pct, tiie_91_pct):
    """
    Returns MXN annualized rate in decimal form.
    Rule set for MVP:
    - <= 28d -> use TIIE 28d
    - >= 91d -> use TIIE 91d
    - otherwise linearly interpolate
    """
    r28 = tiie_28_pct / 100.0
    r91 = tiie_91_pct / 100.0

    if target_days <= 28:
        return r28
    if target_days >= 91:
        return r91

    return linear_interpolate(target_days, 28, r28, 91, r91)


def synthetic_forward(spot, r_mxn, r_usd, days, basis=360):
    tau = days / basis
    return spot * ((1 + r_mxn * tau) / (1 + r_usd * tau))


# ----------------------------
# Banxico / FRED fetchers
# ----------------------------
@st.cache_data(ttl=3600)
def fetch_banxico_series_full(series_id: str):
    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos"
    headers = {"Bmx-Token": BANXICO_API_TOKEN}

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    payload = response.json()

    if "bmx" not in payload or "series" not in payload["bmx"]:
        raise ValueError(f"Unexpected Banxico response for series {series_id}")

    series_list = payload["bmx"]["series"]
    if not series_list:
        raise ValueError(f"No Banxico series returned for {series_id}")

    series = series_list[0]
    datos = series.get("datos", [])

    latest_valid = None
    for entry in reversed(datos):
        dato = str(entry.get("dato", "")).strip()
        if dato not in ["N/E", "", "null", "None"]:
            latest_valid = entry
            break

    if latest_valid is None:
        raise ValueError(f"No valid Banxico datapoints found for {series_id}")

    value = float(str(latest_valid["dato"]).replace(",", ""))

    return {
        "series_id": series_id,
        "title": series.get("titulo", series_id),
        "date": latest_valid.get("fecha"),
        "value": value,
        "raw_data": datos,
    }


@st.cache_data(ttl=3600)
def fetch_fred_latest(series_id: str):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 10,
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    payload = response.json()
    observations = payload.get("observations", [])

    if not observations:
        raise ValueError(f"No FRED observations returned for {series_id}")

    latest_valid = None
    for obs in observations:
        val = str(obs.get("value", "")).strip()
        if val not in [".", "", "null", "None"]:
            latest_valid = obs
            break

    if latest_valid is None:
        raise ValueError(f"No valid FRED datapoints found for {series_id}")

    return {
        "series_id": series_id,
        "date": latest_valid["date"],
        "value": float(latest_valid["value"]) / 100.0,  # decimal
    }


@st.cache_data(ttl=3600)
def fetch_banxico_fix_history():
    data = fetch_banxico_series_full("SF43718")
    rows = []

    for entry in data["raw_data"]:
        dato = str(entry.get("dato", "")).strip()
        fecha = entry.get("fecha")

        if dato in ["N/E", "", "null", "None"]:
            continue

        try:
            spot_value = float(dato.replace(",", ""))
        except Exception:
            continue

        parsed_date = pd.to_datetime(fecha, dayfirst=True, errors="coerce")
        if pd.isna(parsed_date):
            parsed_date = pd.to_datetime(fecha, errors="coerce")

        if pd.isna(parsed_date):
            continue

        rows.append(
            {
                "date": parsed_date,
                "spot": spot_value,
                "pair": "USD/MXN",
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    return df


def get_pair_history(base_df, foreign_currency, local_currency):
    """
    base_df comes from Banxico USD/MXN FIX history.
    Supports:
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
    "Illustrative tool only. Synthetic forward uses Banxico FIX, Banxico TIIE and FRED SOFR."
)

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.header("Tell us about your exposure")

today = pd.Timestamp.now().normalize()
default_settlement = next_business_day(today + pd.Timedelta(days=30))

with st.sidebar.form("hedge_form"):
    transaction_direction = st.radio(
        "My business will...",
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
        f"My business will **{transaction_direction}** **{foreign_currency}** on "
        f"**{pd.Timestamp(settlement_date_input).strftime('%Y-%m-%d')}**, "
        f"and my business operates in **{local_currency}**."
    )

    notional = st.number_input(
        f"Notional amount ({foreign_currency})",
        min_value=1000.0,
        value=1_000_000.0,
        step=1000.0
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
calendar_days = (settlement_dt - today).days

if calendar_days <= 0:
    st.warning("Settlement date must be in the future.")
    st.stop()

# ----------------------------
# Market data load
# ----------------------------
with st.spinner("Fetching Banxico and FRED data..."):
    try:
        banxico_fix = fetch_banxico_series_full("SF43718")
        tiie_28 = fetch_banxico_series_full("SF60648")
        tiie_91 = fetch_banxico_series_full("SF60649")
        sofr = fetch_fred_latest("SOFR")
        usd_mxn_df = fetch_banxico_fix_history()
    except Exception as e:
        st.error(f"Market data fetch failed: {e}")
        st.stop()

if usd_mxn_df.empty:
    st.error("No Banxico FIX history could be built.")
    st.stop()

spot_df = get_pair_history(usd_mxn_df, foreign_currency, local_currency)

if spot_df.empty:
    st.error("This version currently supports only USD/MXN and MXN/USD.")
    st.stop()

# ----------------------------
# Synthetic forward calculation
# ----------------------------
spot_reference = float(banxico_fix["value"])

# Rates for USD/MXN convention
mxn_rate = interpolate_mxn_rate(calendar_days, tiie_28["value"], tiie_91["value"])
usd_rate = float(sofr["value"])  # already decimal

synthetic_forward_usd_mxn = synthetic_forward(
    spot=spot_reference,
    r_mxn=mxn_rate,
    r_usd=usd_rate,
    days=calendar_days,
    basis=360,
)

# Handle MXN/USD if user flips pair
if foreign_currency == "USD" and local_currency == "MXN":
    forward_rate = synthetic_forward_usd_mxn
    spot_display = spot_reference
    forward_points = synthetic_forward_usd_mxn - spot_reference
else:
    forward_rate = 1 / synthetic_forward_usd_mxn
    spot_display = 1 / spot_reference
    forward_points = forward_rate - spot_display

# ----------------------------
# Market data summary
# ----------------------------
st.header("Market Data Snapshot")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Banxico FIX (USD/MXN)", f"{banxico_fix['value']:.4f}")
    st.caption(f"Date: {banxico_fix['date']}")

with col2:
    st.metric("TIIE 28d", f"{tiie_28['value']:.4f}%")
    st.caption(f"Date: {tiie_28['date']}")

with col3:
    st.metric("TIIE 91d", f"{tiie_91['value']:.4f}%")
    st.caption(f"Date: {tiie_91['date']}")

with col4:
    st.metric("SOFR", f"{sofr['value'] * 100:.4f}%")
    st.caption(f"Date: {sofr['date']}")

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
    f"This illustrative forward is derived from **Banxico FIX spot**, **Banxico TIIE**, and **FRED SOFR**."
)

# ----------------------------
# Pricing details
# ----------------------------
st.subheader("Pricing Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Spot Reference", f"{spot_display:.4f}")

with col2:
    st.metric(f"Synthetic Forward ({pair_label})", f"{forward_rate:.4f}")

with col3:
    st.metric("Forward Points", f"{forward_points:.4f}")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Calendar Days to Maturity", int(calendar_days))

with col5:
    st.metric("Implied MXN Rate", f"{mxn_rate * 100:.4f}%")

with col6:
    st.metric("USD Rate Proxy", f"{usd_rate * 100:.4f}%")

# ----------------------------
# Simulation logic
# ----------------------------
TRADING_DAYS_PER_YEAR = 252
tenor_trading_days = max(1, int(calendar_days * (TRADING_DAYS_PER_YEAR / 365)))

if tenor_trading_days >= len(spot_df):
    st.error(
        f"Not enough historical data ({len(spot_df)} records) to simulate "
        f"{tenor_trading_days} trading days."
    )
    st.stop()

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
    st.metric("Effective Trading Days", int(tenor_trading_days))

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Notional", f"{format_notional(notional)} {foreign_currency}")

with col5:
    st.metric(f"Forward Rate Used ({pair_label})", f"{forward_rate:.4f}")

with col6:
    st.metric("Scenario Source", "Banxico FIX history")

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
# Diagnostics
# ----------------------------
with st.expander("Show pricing diagnostics"):
    st.write("Banxico FIX latest:", banxico_fix)
    st.write("Banxico TIIE 28d latest:", tiie_28)
    st.write("Banxico TIIE 91d latest:", tiie_91)
    st.write("FRED SOFR latest:", sofr)
    st.write("Spot reference used:", spot_display)
    st.write("Synthetic forward used:", forward_rate)
    st.write("Forward points:", forward_points)
    st.write("Implied MXN rate:", mxn_rate)
    st.write("USD rate proxy:", usd_rate)

with st.expander("Show raw simulation data"):
    st.dataframe(results_df)

# ----------------------------
# CSV export
# ----------------------------
st.download_button(
    label="Download simulation results (CSV)",
    data=results_df.to_csv(index=False),
    file_name="fx_simulation_results.csv",
    mime="text/csv"
)
