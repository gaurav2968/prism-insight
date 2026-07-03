"""
India (NSE) Stock Data Client

Unified interface for fetching Indian stock market data using nsetools + yfinance.

Usage:
    from prism_in.cores.in_data_client import INDataClient

    client = INDataClient()

    # Get OHLCV data
    df = client.get_ohlcv("RELIANCE", period="1mo")

    # Get company info
    info = client.get_company_info("RELIANCE")

    # Get financials
    financials = client.get_financials("RELIANCE")

Data Sources:
    - nsetools: Real-time quotes, index data, top movers
    - yfinance: Historical OHLCV, fundamentals (uses .NS suffix)
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _yf_ticker(ticker: str) -> str:
    """Convert NSE ticker to yfinance format (add .NS suffix)."""
    if ticker.startswith("^"):
        return ticker  # Index tickers already in yfinance format
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        return f"{ticker}.NS"
    return ticker


class INDataClient:
    """
    Unified India stock data client.

    Provides access to:
    - OHLCV data (yfinance with .NS suffix)
    - Company information (yfinance + nsetools)
    - Financial statements (yfinance)
    - Institutional holders (yfinance)
    - Market cap (nsetools bulk API)
    """

    def __init__(self):
        """Initialize the India data client."""
        self._nse = None

    def _get_nse(self):
        """Lazy-load nsetools Nse instance."""
        if self._nse is None:
            try:
                from nsetools import Nse
                self._nse = Nse()
            except ImportError:
                logger.warning("nsetools not installed. Install with: pip install nsetools")
                self._nse = None
        return self._nse

    # =========================================================================
    # OHLCV Data (yfinance)
    # =========================================================================

    def get_ohlcv(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            ticker: NSE stock ticker (e.g., "RELIANCE", "TCS", "INFY")
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo)
            start: Start date (YYYY-MM-DD), overrides period
            end: End date (YYYY-MM-DD), overrides period

        Returns:
            DataFrame with OHLCV data
        """
        import yfinance as yf

        try:
            yf_sym = _yf_ticker(ticker)
            stock = yf.Ticker(yf_sym)

            if start and end:
                df = stock.history(start=start, end=end, interval=interval)
            else:
                df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No OHLCV data found for {ticker} ({yf_sym})")
                return pd.DataFrame()

            # Standardize column names
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]

            logger.info(f"Retrieved {len(df)} OHLCV records for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {ticker}: {e}")
            return pd.DataFrame()

    def get_stock_ohlcv(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get OHLCV data for a specific date range.

        Args:
            ticker: NSE stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV data
        """
        return self.get_ohlcv(ticker, start=start_date, end=end_date)

    # =========================================================================
    # Company Information (yfinance + nsetools)
    # =========================================================================

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive company information.

        Combines yfinance .info with nsetools quote data.

        Args:
            ticker: NSE stock ticker

        Returns:
            Dictionary with company information
        """
        import yfinance as yf

        try:
            yf_sym = _yf_ticker(ticker)
            stock = yf.Ticker(yf_sym)
            info = stock.info

            if not info or info.get("regularMarketPrice") is None:
                logger.warning(f"No company info found for {ticker}")
                return {}

            # Supplement with nsetools data if available
            nse = self._get_nse()
            if nse:
                try:
                    nse_quote = nse.get_quote(ticker)
                    if nse_quote:
                        info["nse_open"] = nse_quote.get("open", 0)
                        info["nse_dayHigh"] = nse_quote.get("dayHigh", 0)
                        info["nse_dayLow"] = nse_quote.get("dayLow", 0)
                        info["nse_previousClose"] = nse_quote.get("previousClose", 0)
                        info["nse_totalTradedVolume"] = nse_quote.get("totalTradedVolume", 0)
                        info["nse_totalTradedValue"] = nse_quote.get("totalTradedValue", 0)
                        info["nse_deliveryToTradedQuantity"] = nse_quote.get("deliveryToTradedQuantity", 0)
                except Exception as e:
                    logger.debug(f"nsetools supplement failed for {ticker}: {e}")

            logger.info(f"Retrieved company info for {ticker}")
            return info

        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {e}")
            return {}

    # =========================================================================
    # Financial Statements (yfinance)
    # =========================================================================

    def get_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements.

        Args:
            ticker: NSE stock ticker

        Returns:
            Dictionary with income_stmt, balance_sheet, cash_flow DataFrames
        """
        import yfinance as yf

        try:
            yf_sym = _yf_ticker(ticker)
            stock = yf.Ticker(yf_sym)

            result = {
                "income_stmt": stock.income_stmt if stock.income_stmt is not None else pd.DataFrame(),
                "quarterly_income_stmt": stock.quarterly_income_stmt if stock.quarterly_income_stmt is not None else pd.DataFrame(),
                "balance_sheet": stock.balance_sheet if stock.balance_sheet is not None else pd.DataFrame(),
                "quarterly_balance_sheet": stock.quarterly_balance_sheet if stock.quarterly_balance_sheet is not None else pd.DataFrame(),
                "cash_flow": stock.cashflow if stock.cashflow is not None else pd.DataFrame(),
                "quarterly_cash_flow": stock.quarterly_cashflow if stock.quarterly_cashflow is not None else pd.DataFrame(),
            }

            logger.info(f"Retrieved financials for {ticker}")
            return result

        except Exception as e:
            logger.error(f"Error fetching financials for {ticker}: {e}")
            return {}

    # =========================================================================
    # Holder Information (yfinance)
    # =========================================================================

    def get_institutional_holders(self, ticker: str) -> Dict[str, Any]:
        """
        Get holder information.

        Args:
            ticker: NSE stock ticker

        Returns:
            Dictionary with major_holders, institutional_holders, mutualfund_holders
        """
        import yfinance as yf

        try:
            yf_sym = _yf_ticker(ticker)
            stock = yf.Ticker(yf_sym)

            result = {}

            try:
                mh = stock.major_holders
                if mh is not None and not mh.empty:
                    result["major_holders"] = mh
            except Exception:
                pass

            try:
                ih = stock.institutional_holders
                if ih is not None and not ih.empty:
                    result["institutional_holders"] = ih
            except Exception:
                pass

            try:
                mfh = stock.mutualfund_holders
                if mfh is not None and not mfh.empty:
                    result["mutualfund_holders"] = mfh
            except Exception:
                pass

            logger.info(f"Retrieved holder info for {ticker}: {list(result.keys())}")
            return result

        except Exception as e:
            logger.error(f"Error fetching holder info for {ticker}: {e}")
            return {}

    # =========================================================================
    # Recommendations (yfinance)
    # =========================================================================

    def get_recommendations(self, ticker: str) -> pd.DataFrame:
        """
        Get analyst recommendations.

        Args:
            ticker: NSE stock ticker

        Returns:
            DataFrame with recommendations
        """
        import yfinance as yf

        try:
            yf_sym = _yf_ticker(ticker)
            stock = yf.Ticker(yf_sym)
            recs = stock.recommendations
            if recs is not None and not recs.empty:
                logger.info(f"Retrieved {len(recs)} recommendations for {ticker}")
                return recs
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching recommendations for {ticker}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # Real-time Quote (nsetools)
    # =========================================================================

    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get real-time quote from NSE.

        Args:
            ticker: NSE stock ticker

        Returns:
            Dictionary with quote data
        """
        nse = self._get_nse()
        if nse is None:
            logger.warning("nsetools not available for real-time quote")
            return {}

        try:
            return nse.get_quote(ticker)
        except Exception as e:
            logger.error(f"Error fetching quote for {ticker}: {e}")
            return {}

    # =========================================================================
    # Index Data
    # =========================================================================

    def get_index_data(self, index_name: str = "NIFTY 50") -> Dict[str, Any]:
        """
        Get index quote data.

        Args:
            index_name: Index name (e.g., "NIFTY 50", "NIFTY BANK")

        Returns:
            Dictionary with index data
        """
        nse = self._get_nse()
        if nse is None:
            return {}

        try:
            return nse.get_index_quote(index_name)
        except Exception as e:
            logger.error(f"Error fetching index data for {index_name}: {e}")
            return {}

    def get_index_ohlcv(
        self,
        index_name: str = "NIFTY 50",
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Get historical index OHLCV data via yfinance.

        Args:
            index_name: Index name
            period: Data period

        Returns:
            DataFrame with OHLCV data
        """
        import yfinance as yf

        # Map NSE index names to yfinance tickers
        yf_map = {
            "NIFTY 50": "^NSEI",
            "NIFTY BANK": "^NSEBANK",
            "NIFTY IT": "^CNXIT",
            "NIFTY PHARMA": "^CNXPHARMA",
            "NIFTY AUTO": "^CNXAUTO",
            "NIFTY FMCG": "^CNXFMCG",
            "NIFTY METAL": "^CNXMETAL",
            "NIFTY ENERGY": "^CNXENERGY",
            "NIFTY MIDCAP 50": "^NSEMDCP50",
            "BSE SENSEX": "^BSESN",
        }

        yf_ticker = yf_map.get(index_name, "^NSEI")

        try:
            stock = yf.Ticker(yf_ticker)
            df = stock.history(period=period)
            if df.empty:
                logger.warning(f"No index OHLCV data for {index_name}")
                return pd.DataFrame()
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            return df
        except Exception as e:
            logger.error(f"Error fetching index OHLCV for {index_name}: {e}")
            return pd.DataFrame()

    # =========================================================================
    # Market Cap
    # =========================================================================

    def get_market_cap(self, ticker: str) -> float:
        """
        Get market capitalization in INR.

        Args:
            ticker: NSE stock ticker

        Returns:
            Market cap in INR (0 if unavailable)
        """
        info = self.get_company_info(ticker)
        return float(info.get("marketCap", 0) or 0)


if __name__ == "__main__":
    """Quick test of the India data client."""
    logging.basicConfig(level=logging.INFO)

    client = INDataClient()

    print("\n=== Testing INDataClient ===\n")

    # Test OHLCV
    print("1. OHLCV for RELIANCE (1mo):")
    df = client.get_ohlcv("RELIANCE", period="1mo")
    if not df.empty:
        print(f"   {len(df)} records, latest close: ₹{df['close'].iloc[-1]:.2f}")
    else:
        print("   No data")

    # Test company info
    print("\n2. Company info for TCS:")
    info = client.get_company_info("TCS")
    if info:
        print(f"   Name: {info.get('longName', 'N/A')}")
        print(f"   Price: ₹{info.get('regularMarketPrice', 'N/A')}")
        print(f"   Market Cap: ₹{info.get('marketCap', 0):,.0f}")
    else:
        print("   No data")

    # Test index data
    print("\n3. NIFTY 50 index:")
    idx = client.get_index_ohlcv("NIFTY 50", period="5d")
    if not idx.empty:
        print(f"   Latest close: {idx['close'].iloc[-1]:.2f}")
    else:
        print("   No data")

    print("\n=== Test Complete ===")
