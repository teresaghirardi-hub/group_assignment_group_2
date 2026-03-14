"""
pysimfin.py
-----------
Python API wrapper for SimFin (Part 2.1).
"""

import time
import requests
import pandas as pd


class SimFinAPIError(Exception):
    pass

class SimFinRateLimitError(SimFinAPIError):
    pass

class SimFinNotFoundError(SimFinAPIError):
    pass


class PySimFin:
    """Python wrapper for the SimFin REST API."""

    BASE_URL      = "https://backend.simfin.com/api/v3"
    REQUEST_DELAY = 0.6

    def __init__(self, api_key: str):
        if not api_key or not isinstance(api_key, str):
            raise ValueError("A valid SimFin API key string is required.")
        self._api_key           = api_key
        self._headers           = {
            "Authorization": f"api-key {api_key}",
            "Content-Type":  "application/json",
        }
        self._last_request_time = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: dict) -> list:
        self._throttle()
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = requests.get(url, headers=self._headers, params=params, timeout=15)
        except requests.exceptions.ConnectionError:
            raise SimFinAPIError("Could not connect to SimFin.")
        except requests.exceptions.Timeout:
            raise SimFinAPIError("SimFin request timed out.")
        if response.status_code == 429:
            raise SimFinRateLimitError("Rate limit hit. Wait and retry.")
        if response.status_code == 404:
            raise SimFinNotFoundError(f"Not found: {endpoint}")
        if response.status_code == 401:
            raise SimFinAPIError("Invalid API key.")
        if response.status_code != 200:
            raise SimFinAPIError(f"SimFin error {response.status_code}: {response.text}")
        return response.json()

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if not ticker:
            raise ValueError("ticker must be a non-empty string.")
        if start > end:
            raise ValueError(f"start ({start}) must be before end ({end}).")

        params = {"ticker": ticker.upper(), "start": start, "end": end}

        try:
            data = self._get("companies/prices/compact", params)
        except SimFinNotFoundError:
            raise SimFinNotFoundError(f"Ticker '{ticker}' not found in SimFin.")

        if not data or not isinstance(data, list):
            return pd.DataFrame()

        company = data[0]
        if "data" not in company or "columns" not in company:
            raise SimFinAPIError(f"Unexpected response format for {ticker}.")

        df = pd.DataFrame(company["data"], columns=company["columns"])

        df.columns = [
            c.lower().replace(" ", "_").replace(".", "_").replace("/", "_")
            for c in df.columns
        ]

        COLUMN_MAP = {
            "last_closing_price":        "close",
            "highest_price":             "high",
            "lowest_price":              "low",
            "opening_price":             "open",
            "adjusted_closing_price":    "adj_close",
            "trading_volume":            "volume",
            "common_shares_outstanding": "shares_outstanding",
            "dividend_paid":             "dividend",
        }
        df = df.rename(columns=COLUMN_MAP)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def get_financial_statement(
        self, ticker: str, start: str, end: str,
        statement: str = "pl", period: str = "annual",
    ) -> pd.DataFrame:
        valid = {"pl", "bs", "cf"}
        if statement not in valid:
            raise ValueError(f"statement must be one of {valid}.")
        params = {"ticker": ticker.upper(), "start": start, "end": end,
                  "statement": statement, "period": period}
        try:
            data = self._get("companies/statements/compact", params)
        except SimFinNotFoundError:
            raise SimFinNotFoundError(f"Financial statement for '{ticker}' not found.")
        if not data or not isinstance(data, list):
            return pd.DataFrame()
        company = data[0]
        if "data" not in company or "columns" not in company:
            raise SimFinAPIError(f"Unexpected response format for {ticker}.")
        df = pd.DataFrame(company["data"], columns=company["columns"])
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df

    def get_company_info(self, ticker: str) -> dict:
        params = {"ticker": ticker.upper()}
        try:
            data = self._get("companies/general/compact", params)
        except SimFinNotFoundError:
            raise SimFinNotFoundError(f"Company '{ticker}' not found.")
        if not data or not isinstance(data, list):
            return {}
        company = data[0]
        if "data" not in company or "columns" not in company:
            return {}
        return dict(zip(company["columns"], company["data"][0]))