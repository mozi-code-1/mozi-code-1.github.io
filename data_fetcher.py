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
        end_date: str,
        covid_handling: str = "No adjustment"
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Fetch all macroeconomic variables for a country

        Args:
            country: Country name
            start_date: Start date for data
            end_date: End date for data
            covid_handling: COVID-19 handling method

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
                    # Only A191RL1Q225SBEA (US) is already a growth rate
                    # All other series are GDP levels and need to be converted
                    if series_id == 'A191RL1Q225SBEA':
                        # US series is already annualized growth rate
                        pass
                    else:
                        # Convert levels to quarter-over-quarter growth rate (annualized)
                        series_q = series_q.pct_change() * 100 * 4
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

        # Drop rows with any NaN values (from transformations)
        df = df.dropna()

        # Rename columns for clarity
        df.columns = ['GDP Growth', 'Unemployment', 'Inflation', 'Interest Rate']

        # Align date ranges - find common dates across all variables
        if len(df) > 0:
            # Ensure we have complete data
            df = df.dropna()

        # Handle COVID-19 period based on user selection
        if covid_handling == "Exclude COVID period":
            # Drop 2020Q1 through 2021Q4
            covid_start = pd.Timestamp('2020-01-01')
            covid_end = pd.Timestamp('2021-12-31')
            df = df[(df.index < covid_start) | (df.index > covid_end)]
        elif covid_handling == "COVID dummy variable":
            # Add a COVID dummy column (will be used as exogenous variable)
            covid_start = pd.Timestamp('2020-01-01')
            covid_end = pd.Timestamp('2021-12-31')
            df['COVID_DUMMY'] = ((df.index >= covid_start) & (df.index <= covid_end)).astype(int)

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


def align_date_ranges(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two dataframes to have the same date range

    Args:
        df1: First dataframe
        df2: Second dataframe

    Returns:
        Tuple of aligned dataframes with common date range
    """
    # Find common date range
    common_start = max(df1.index.min(), df2.index.min())
    common_end = min(df1.index.max(), df2.index.max())

    # Filter both dataframes to common range
    df1_aligned = df1[(df1.index >= common_start) & (df1.index <= common_end)]
    df2_aligned = df2[(df2.index >= common_start) & (df2.index <= common_end)]

    # Find exact common dates (in case of missing values)
    common_dates = df1_aligned.index.intersection(df2_aligned.index)

    df1_aligned = df1_aligned.loc[common_dates]
    df2_aligned = df2_aligned.loc[common_dates]

    return df1_aligned, df2_aligned
