# RL Stock Trading Agent (Indian Markets) 📈🤖

An end-to-end **Reinforcement Learning** stock trading application for Indian equities (NSE / BSE), with **6+ RL algorithms** to compare, **Azure OpenAI / Groq** powered insights, and a **Streamlit** dashboard.

> Built as a **learning project**: every algorithm is documented in plain English, and the codebase is modular so you can swap pieces in/out.

---

## ✨ Features

- 📊 **Data**: Free historical NSE/BSE data via `yfinance`
- 🧠 **6 RL Algorithms**: DQN, PPO, A2C, SAC, TD3, DDPG (+ Double/Dueling DQN)
- 🏟️ **Custom Gymnasium environment** for stock trading
- 📈 **Technical indicators**: RSI, MACD, Bollinger Bands, SMA/EMA, ATR, OBV, ADX
- 🔬 **MLflow experiment tracking**
- 🤖 **AI Insights** (Azure OpenAI primary, Groq fallback):
  - Market sentiment analysis
  - Decision explanations in plain English
  - Strategy recommendations
  - Portfolio performance summaries
- 🖥️ **Streamlit dashboard** with interactive charts
- ✅ **Pytest** test suite

---

## 🚀 Quick Start

### 1. Install dependencies
```powershell
.\myenv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment
```powershell
copy .env.example .env
# Edit .env with your Azure OpenAI or Groq keys
```

### 3. Fetch data
```powershell
python -m data.fetch_data
```

### 4. Train an agent
```powershell
python -m training.trainer --algo PPO --ticker RELIANCE.NS --timesteps 50000
```

### 5. Compare all agents
```powershell
python -m evaluation.comparison --ticker RELIANCE.NS
```

### 6. Launch dashboard
```powershell
streamlit run dashboard/app.py
```

---

## 📚 Documentation

All in [.github/](./.github/):
- [ARCHITECTURE.md](./.github/ARCHITECTURE.md) — System design overview
- [ALGORITHMS.md](./.github/ALGORITHMS.md) — Each RL algorithm explained simply
- [SETUP.md](./.github/SETUP.md) — Detailed installation guide
- [LEARNING_GUIDE.md](./.github/LEARNING_GUIDE.md) — Use this project to learn RL
- [CONTRIBUTING.md](./.github/CONTRIBUTING.md) — Contribution guidelines

---

## 🗂️ Project Layout

```
config/         # Pydantic-based configuration
data/           # Data fetching scripts
features/       # Technical indicators
environments/   # Custom Gym trading env + reward strategies
agents/         # All RL agent implementations
training/       # Training pipeline + hyperparameters
evaluation/     # Backtesting, metrics, comparison
insights/       # LLM client + AI-powered insights
dashboard/      # Streamlit multi-page app
tests/          # Pytest test suite
notebooks/      # Jupyter walkthroughs
```

---

## ⚠️ Disclaimer

This is an **educational project**. Do **NOT** use it for live trading with real money. Past performance on historical data does not guarantee future results.

---

## 📜 License

MIT
