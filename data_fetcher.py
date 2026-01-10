"""
Data fetching module for macroeconomic data from FRED
"""
import pandas as pd
import numpy as np
from fredapi import Fred
from typing import Dict, Tuple, Optional
import streamlit as st


# Country-specific FRED series codes
COUNTRY_SERIES = {
    'United States': {
        'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP growth rate
        'unemployment': 'UNRATE',  # Unemployment rate
        'inflation': 'CPIAUCSL',  # CPI
        'interest_rate': 'DGS3MO',  # 3-month Treasury
    },
    'United Kingdom': {
        'gdp_growth': 'GBRRGDPQDSNAQ',  # Real GDP growth
        'unemployment': 'LRUNTTTTGBQ156S',  # Unemployment rate
        'inflation': 'GBRCPIALLMINMEI',  # CPI
        'interest_rate': 'IR3TIB01GBQ156N',  # 3-month rate
    },
    'Germany': {
        'gdp_growth': 'CLVMNACSCAB1GQDE',  # Real GDP
        'unemployment': 'LRUNTTTTDEQ156S',  # Unemployment rate
        'inflation': 'DEUCPIALLMINMEI',  # CPI
        'interest_rate': 'IR3TIB01DEM156N',  # 3-month rate
    },
    'Japan': {
        'gdp_growth': 'JPNRGDPEXP',  # Real GDP
        'unemployment': 'LRUNTTTTJPQ156S',  # Unemployment rate
        'inflation': 'JPNCPIALLMINMEI',  # CPI
        'interest_rate': 'IR3TIB01JPQ156N',  # 3-month rate
    },
    'Canada': {
        'gdp_growth': 'NGDPRSAXDCCAQ',  # Real GDP
        'unemployment': 'LRUNTTTTCAQ156S',  # Unemployment rate
        'inflation': 'CANCPIALLMINMEI',  # CPI
        'interest_rate': 'IR3TIB01CAQ156N',  # 3-month rate
    },
    'France': {
        'gdp_growth': 'CLVMNACSCAB1GQFR',  # Real GDP
        'unemployment': 'LRUNTTTTFRQ156S',  # Unemployment rate
        'inflation': 'FRACPIALLMINMEI',  # CPI
        'interest_rate': 'IR3TIB01FRQ156N',  # 3-month rate
    },
    'Italy': {
        'gdp_growth': 'CLVMNACSCAB1GQIT',  # Real GDP
        'unemployment': 'LRUNTTTTITQ156S',  # Unemployment rate
        'inflation': 'ITACPIALLMINMEI',  # CPI
        'interest_rate': 'IR3TIB01ITQ156N',  # 3-month rate
    },
    'Australia': {
        'gdp_growth': 'AUSRGDPEXP',  # Real GDP
        'unemployment': 'LRUNTTTTAUQ156S',  # Unemployment rate
        'inflation': 'AUSCPIALLQINMEI',  # CPI
        'interest_rate': 'IR3TIB01AUQ156N',  # 3-month rate
    },
}


class DataFetcher:
    def __init__(self, api_key: str):
        """Initialize FRED API connection"""
        self.fred = Fred(api_key=api_key)

    def fetch_country_data(
        self,
        country: str,
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Fetch all macroeconomic variables for a country

        Returns:
            data: DataFrame with all variables
            series_info: Dict with metadata about each series
        """
        if country not in COUNTRY_SERIES:
            raise ValueError(f"Country {country} not supported")

        series_codes = COUNTRY_SERIES[country]
        data_dict = {}
        series_info = {}

        for var_name, series_id in series_codes.items():
            try:
                series = self.fred.get_series(series_id, start_date, end_date)

                # Convert to quarterly if needed
                if series.index.freq is None:
                    series.index = pd.to_datetime(series.index)

                # Resample to quarterly, taking last value of quarter
                series_q = series.resample('Q').last()

                # Handle specific transformations
                if var_name == 'gdp_growth':
                    # If we have GDP levels, convert to growth rates
                    if series_id not in ['A191RL1Q225SBEA', 'GBRRGDPQDSNAQ']:
                        series_q = series_q.pct_change() * 100
                elif var_name == 'inflation':
                    # Convert CPI to inflation rate (YoY % change)
                    series_q = series_q.pct_change(4) * 100

                data_dict[var_name] = series_q

                # Get series information
                info = self.fred.get_series_info(series_id)
                series_info[var_name] = {
                    'title': info['title'],
                    'units': info['units'],
                    'frequency': info['frequency']
                }

            except Exception as e:
                st.warning(f"Could not fetch {var_name} for {country}: {str(e)}")
                # Create empty series
                data_dict[var_name] = pd.Series(dtype=float)

        # Combine into DataFrame
        df = pd.DataFrame(data_dict)

        # Drop rows with any NaN values
        df = df.dropna()

        # Rename columns for clarity
        df.columns = ['GDP Growth', 'Unemployment', 'Inflation', 'Interest Rate']

        return df, series_info

    def get_available_countries(self) -> list:
        """Return list of available countries"""
        return list(COUNTRY_SERIES.keys())


def calculate_growth_rate(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate percentage growth rate"""
    return series.pct_change(periods) * 100


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics for the data"""
    summary = df.describe()

    # Add additional statistics
    summary.loc['skewness'] = df.skew()
    summary.loc['kurtosis'] = df.kurtosis()

    return summary
