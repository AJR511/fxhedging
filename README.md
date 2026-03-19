# Yarda FX Hedging Simulator

An interactive web application to simulate FX hedging strategies using historical USD/MXN data.

Built as part of Yarda, a platform focused on helping Latin American companies better understand and manage foreign exchange risk.

---

## Overview

This application allows users to:

- Simulate FX outcomes based on approximately 20 years of historical USD/MXN data
- Compare unhedged and hedged strategies
- Analyze PnL distributions across different hedge ratios
- Evaluate best-case, worst-case, and average outcomes

It is designed to help CFOs and finance teams visualize the impact of hedging decisions before executing trades.

---

## Methodology

The model follows these steps:

1. Retrieves historical USD/MXN daily data from OANDA  
2. Constructs rolling historical scenarios based on the selected tenor  
3. Applies those scenarios to a user-defined forward rate  
4. Simulates outcomes across different hedge ratios (0% to 100%)  
5. Produces summary statistics and distribution visualizations  

---

## Inputs

- Transaction direction (Pay / Receive)  
- Notional amount (USD)  
- Settlement date  
- Forward rate (USD/MXN)  
- Hedge ratios (0% to 100%)  

---

## Outputs

- PnL summary table (best, worst, average)  
- Distribution of outcomes across hedging strategies  
- Visual comparison of hedging effectiveness  

---

## Tech Stack

- Streamlit (application framework)  
- Python (pandas) for data processing  
- OANDA API for historical FX data  
- Matplotlib and Seaborn for visualization  

---

## Secrets

This application requires an OANDA API key.

When deploying on Streamlit Cloud, add the following in the Secrets manager:

```toml
OANDA_API_KEY = "your_api_key_here"
