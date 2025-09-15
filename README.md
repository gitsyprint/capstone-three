# 29.4.1
# Capstone Two Final Report: Integrating Price Signals and LLM Sentiment for SPY Prediction
## Introduction
This project investigates whether large language models (LLMs) can provide incremental predictive power for financial forecasting. Specifically, it asks: *Can LLM-derived sentiment improve daily predictive models of SPY ETF returns when combined with standard price-based technical features?*
With markets increasingly influenced by real-time news, investors and analysts are turning to language models to mine headlines for tradable signals. The goal of this project was to quantify whether sentiment extracted from financial news using FinBERT, a finance-tuned transformer, could improve upon traditional technical indicators. By testing predictive models both with and without sentiment features, the project evaluated the standalone and incremental predictive value of LLM-derived news sentiment at the daily frequency.
## Data and Methods
Daily price and volume data for SPY from 2015 onward were collected via Yahoo Finance. News headlines mentioning SPY (with fallback to AAPL when coverage was sparse) were retrieved from Yahoo Finance and scored using FinBERT. This model provided per-headline sentiment probabilities (negative, neutral, positive) as well as a signed score defined as positive minus negative.
Price data were normalized into a flat OHLCV schema, and dates were deduplicated to ensure clean alignment with headlines. Both raw and processed datasets were stored for reproducibility. To synchronize signals, prices and headlines were merged on calendar date, avoiding leakage from timezone mismatches.
Two categories of features were constructed. Price-based features included daily percent returns, a 20-day simple moving average, and Ichimoku Cloud components (tenkan, kijun, spans A and B). Sentiment features were aggregated daily by averaging headline-level probabilities, then lagged and smoothed to prevent forward-looking bias. Target variables included next-day returns for regression and an up/down classification indicator.
All model inputs were standardized using scikit-learn’s *StandardScaler*, ensuring comparability across features. An 80/20 chronological split preserved the time order of data, with the final 20% reserved for out-of-sample evaluation.
Three families of models were trained to satisfy rubric requirements for methodological diversity:
•	**Linear and Logistic Regression** as baseline models
•	**XGBoost Regressor and Classifier** as nonlinear methods
•	**Logistic Regression with sentiment** integration as a hybrid approach
Each model was tested both with and without FinBERT sentiment features to isolate incremental contribution.
## Exploratory Data Analysis
Exploratory data analysis confirmed well-known characteristics of SPY returns. A histogram of daily returns showed the heavy concentration near zero with fat-tailed extremes, underscoring the inherent difficulty of prediction. Ichimoku Cloud overlays visually distinguished trending versus ranging regimes, while a correlation heatmap quantified weak linear relationships between price-based features and next-day returns. FinBERT sentiment displayed intuitive alignment with market regimes. A five-day rolling average of sentiment scores trended upward during rallies and downward during sell-offs.

*Figure 1. Histogram of SPY daily returns — clustering near zero with fat tails.*
