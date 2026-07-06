# AlphaPulse Dashboard

AI-powered Indian stock market analysis and trading dashboard.

## Dashboard Features

- **Portfolio Overview**: Capital snapshot, current equity, holdings count
- **Historical Trades**: Complete trade history with dates, quantities, gross/net P&L, and fees
- **Recent Trades**: Last 4 closed positions with detailed metrics
- **Market Regime**: Current market conditions and regime classification
- **Trigger Analysis**: Daily stock screening results and regime signals
- **Watchlist**: Historical analysis of stocks not yet entered

## Data Sources

- Stock prices: Yahoo Finance (yfinance)
- Market context: NIFTY 50 technical analysis
- Trade history: AlphaPulse simulator database
- Trigger signals: Daily NSE surge detection

## Refresh Schedule

Dashboard snapshot is automatically regenerated twice daily:
- **9:30 AM IST** (4:00 AM UTC) - Pre-market session
- **3:30 PM IST** (10:00 AM UTC) - Post-market session

## Notes

- P&L calculations include estimated Indian delivery transaction costs (~0.193% round trip)
- Portfolio uses equal-slot simulator capital allocation model
- All prices in Indian Rupees (₹)

---

Powered by **AlphaPulse** | [GitHub](https://github.com/gaurav2968/) | India (NSE) Market

**Creator**: Kumar Gaurav
- 💼 [LinkedIn](https://www.linkedin.com/in/kumar-gaurav-908942150/)
- 📧 rockstar.gs139@gmail.com
