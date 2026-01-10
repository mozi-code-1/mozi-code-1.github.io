# 📊 Interactive Macroeconomic VAR Dashboard

A comprehensive web-based dashboard for comparing macroeconomic dynamics across countries using Vector Autoregression (VAR) and Vector Error Correction Models (VECM).

## 🌟 Features

### Data & Visualization
- **Country Selection**: Choose from 8 major economies (US, UK, Germany, Japan, Canada, France, Italy, Australia)
- **Quarterly Data**: Real GDP growth, unemployment rate, inflation (CPI), and short-term interest rates
- **Flexible Date Ranges**: Adjustable date range slider (1960-present)
- **Interactive Time Series Plots**: View and compare raw data across countries

### Econometric Analysis

#### 1. **Stationarity Testing**
- Augmented Dickey-Fuller (ADF) tests for all variables
- Automatic identification of non-stationary series
- Recommendations for differencing or cointegration testing

#### 2. **Cointegration Analysis**
- Johansen cointegration test with trace and maximum eigenvalue statistics
- Automatic detection of cointegrating relationships
- Switches to VECM when cointegration is present

#### 3. **VAR/VECM Estimation**
- Automatic lag length selection using AIC, BIC, or HQIC
- Support for both VAR (no cointegration) and VECM (cointegration detected)
- Comprehensive model diagnostics:
  - Portmanteau test for residual autocorrelation
  - Normality tests
  - Model summary statistics

#### 4. **Identification Schemes**
Toggle between two structural identification methods:
- **Cholesky Decomposition**: Variables ordered as [GDP Growth → Unemployment → Inflation → Interest Rate]
- **Blanchard-Quah Long-Run Restrictions**: Separates demand vs. supply shocks

#### 5. **Impulse Response Functions (IRFs)**
- IRFs for all variable pairs over customizable horizons (up to 40 quarters)
- Bootstrap confidence bands (68%, 90%, or 95%)
- Interactive selection of shock and response variables
- View individual IRFs or all at once in a grid

#### 6. **Forecasting**
- Multi-step ahead forecasts (up to 20 quarters)
- 95% confidence intervals
- Forecast Error Variance Decomposition (FEVD)
  - Shows contribution of each structural shock to forecast uncertainty
  - Particularly useful for understanding unemployment dynamics

#### 7. **Granger Causality**
- Pairwise Granger causality tests between all variables
- Tests multiple lag orders
- Highlights statistically significant relationships
- Separate results for each country

#### 8. **Historical Decomposition**
- Decompose actual variable values into contributions from each structural shock
- Shows how different shocks (monetary, demand, supply, labor market) contributed to historical movements
- Particularly insightful for understanding unemployment fluctuations

### User Experience
- **Clean, Functional UI**: Tabbed interface for easy navigation
- **CSV Downloads**: Export all results (IRFs, forecasts, decompositions) for external analysis
- **Responsive Design**: Works on desktop and tablet
- **Real-time Updates**: All charts and tables update dynamically

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- A FRED API key (free)

### Step 1: Get a FRED API Key

1. Go to https://fred.stlouisfed.org/
2. Create a free account (click "My Account" → "Register")
3. Once logged in, go to https://fred.stlouisfed.org/docs/api/api_key.html
4. Click "Request API Key"
5. Fill out the simple form and submit
6. Copy your API key (you'll need this to run the dashboard)

### Step 2: Install Python

**Windows:**
1. Download Python from https://www.python.org/downloads/
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH" during installation
4. Click "Install Now"

**Mac:**
1. Python usually comes pre-installed, but for the latest version:
2. Download from https://www.python.org/downloads/
3. Run the installer

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Step 3: Download the Dashboard Files

All files are already in this directory:
   - `app.py` - Main dashboard application
   - `data_fetcher.py` - Data retrieval module
   - `var_models.py` - VAR/VECM estimation module
   - `requirements.txt` - Python dependencies

### Step 4: Install Dependencies

Open a terminal/command prompt and navigate to this folder:

**Windows:**
```bash
cd path\to\mozi-code-1.github.io
python -m pip install -r requirements.txt
```

**Mac/Linux:**
```bash
cd path/to/mozi-code-1.github.io
pip3 install -r requirements.txt
```

This will install all necessary packages (may take 2-3 minutes).

### Step 5: Run the Dashboard

In the same terminal, type:

```bash
streamlit run app.py
```

Your default web browser should automatically open to the dashboard (usually at http://localhost:8501).

If it doesn't open automatically, look for a message in the terminal like:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Copy that URL and paste it into your browser.

## 📖 How to Use

### Basic Workflow

1. **Enter API Key**: In the sidebar, paste your FRED API key

2. **Select Countries**: Choose two countries to compare from the dropdown menus

3. **Set Date Range**: Use the date pickers to select your analysis period
   - Tip: Use at least 20 years of data for reliable results
   - More data = better statistical properties

4. **Configure Model Settings**:
   - **Maximum Lags**: Default is 8, which is good for quarterly data
   - **Information Criterion**: BIC tends to select more parsimonious models
   - **Identification Scheme**: Start with Cholesky decomposition

5. **Fetch Data**: Click "Fetch Data and Estimate Models"
   - The dashboard will retrieve data and automatically run all tests

6. **Explore Tabs**:

   **📈 Data Overview**
   - View raw time series for both countries
   - Check summary statistics
   - Verify data quality

   **🔍 Stationarity & Cointegration**
   - Review ADF test results
   - Check for cointegration
   - Understand whether VAR or VECM is appropriate

   **📊 VAR/VECM Estimation**
   - See optimal lag length selection
   - Review model fit and diagnostics
   - Check residual properties

   **💥 Impulse Responses**
   - Select shock and response variables
   - View dynamic responses over time
   - Interpret confidence bands

   **🔮 Forecasting**
   - Generate out-of-sample forecasts
   - Examine forecast uncertainty
   - Analyze variance decomposition

   **🔗 Granger Causality**
   - Identify predictive relationships
   - Compare causal structures across countries

   **📉 Historical Decomposition**
   - See how shocks contributed to actual data
   - Understand drivers of unemployment or other variables

### Example Analysis

**Research Question**: How do monetary policy shocks affect unemployment in the US vs. UK?

1. Select US and UK as your countries
2. Set date range: 2000-01-01 to 2023-12-31
3. Click "Fetch Data and Estimate Models"
4. Go to **Impulse Responses** tab
5. Set:
   - Shock Variable: "Interest Rate"
   - Response Variable: "Unemployment"
6. Compare the IRFs for both countries
7. Download the data for your paper/presentation

## 🎓 Interpretation Guide

### Reading Impulse Response Functions
- **Positive response**: The response variable increases following the shock
- **Negative response**: The response variable decreases following the shock
- **Persistence**: How long the effect lasts
- **Confidence bands**: If they cross zero, the effect isn't statistically significant

### Variance Decomposition
- Shows what % of forecast error variance is due to each shock
- Example: "40% of unemployment forecast error is due to monetary policy shocks"
- Helps identify the most important drivers of uncertainty

### Granger Causality
- "X Granger-causes Y" means past values of X help predict Y
- Does NOT imply true causality (correlation ≠ causation)
- Useful for understanding predictive relationships

### Model Selection
- **AIC**: Tends to select more lags (better for forecasting)
- **BIC**: Tends to select fewer lags (better for interpretation)
- **HQIC**: Middle ground between AIC and BIC

## 📊 Available Countries & Data

| Country | GDP Growth | Unemployment | Inflation | Interest Rate |
|---------|-----------|--------------|-----------|---------------|
| United States | ✅ | ✅ | ✅ | ✅ |
| United Kingdom | ✅ | ✅ | ✅ | ✅ |
| Germany | ✅ | ✅ | ✅ | ✅ |
| Japan | ✅ | ✅ | ✅ | ✅ |
| Canada | ✅ | ✅ | ✅ | ✅ |
| France | ✅ | ✅ | ✅ | ✅ |
| Italy | ✅ | ✅ | ✅ | ✅ |
| Australia | ✅ | ✅ | ✅ | ✅ |

Data availability varies by country. The dashboard automatically handles missing data.

## 🛠️ Troubleshooting

### "Module not found" error
- Make sure you ran `pip install -r requirements.txt`
- Try `pip3` instead of `pip` on Mac/Linux

### "API key invalid" error
- Double-check you copied the entire API key
- Make sure there are no extra spaces
- Request a new key if needed

### "Insufficient data" warning
- Try a different date range
- Some countries have limited data before certain years
- Use at least 20 quarters (5 years) of data

### Charts not displaying
- Refresh the page
- Clear your browser cache
- Try a different browser (Chrome/Firefox recommended)

### Slow performance
- Reduce the number of bootstrap replications (edit `n_bootstrap` in code)
- Use a shorter time period
- Select fewer maximum lags

## 📚 Technical Details

### Methodology

**Stationarity Testing**
- Augmented Dickey-Fuller (ADF) test with automatic lag selection via AIC
- Null hypothesis: Unit root (non-stationary)
- 5% significance level

**Cointegration Testing**
- Johansen procedure with trace and maximum eigenvalue statistics
- Deterministic terms: Constant in cointegrating equation
- Lag selection based on VAR specification

**VAR Estimation**
- Ordinary Least Squares (OLS) estimation
- Lag selection via information criteria
- Assumes no cointegration or data already differenced

**VECM Estimation**
- Maximum likelihood estimation
- Incorporates cointegrating relationships
- More efficient when cointegration exists

**Identification**
- Cholesky: Recursive structure, ordering matters
- Blanchard-Quah: Long-run restrictions, ordering matters less

**Bootstrap Inference**
- Residual-based bootstrap for confidence bands
- 500-1000 replications (adjustable)
- Percentile method for confidence intervals

### Software Stack
- **Backend**: Python 3.8+
- **Web Framework**: Streamlit 1.31
- **Econometrics**: statsmodels 0.14
- **Numerical Computing**: NumPy, SciPy, pandas
- **Visualization**: Plotly 5.18
- **Data Source**: FRED API (fredapi 0.5)

## 📝 Citation

If you use this dashboard for research, please cite:

```
Macroeconomic VAR Dashboard (2024)
Interactive tool for cross-country VAR/VECM analysis
Data: Federal Reserve Economic Data (FRED)
```

## ⚠️ Disclaimer

This is an educational tool for learning about VAR models and macroeconomic analysis. While the methodology is sound, results should be:
- Validated with additional robustness checks for research
- Interpreted with appropriate domain knowledge
- Not used as the sole basis for policy decisions

The quality of results depends on:
- Data quality and availability
- Appropriate model specification
- Correct interpretation of results

## 🤝 Contributing

Found a bug or want to add a feature? This is a self-contained educational tool, but improvements are welcome:
- More countries/regions
- Additional identification schemes
- Sign restrictions
- Panel VAR capabilities
- Alternative data sources

## 📧 Support

For issues with:
- **FRED API**: Contact FRED support at https://fred.stlouisfed.org/
- **Python/Installation**: Search Stack Overflow or Python documentation
- **Statistical Methods**: Consult econometrics textbooks (e.g., Hamilton, Lütkepohl)

## 📖 Recommended Reading

- **Stock, J. H., & Watson, M. W.** (2001). Vector autoregressions. Journal of Economic perspectives, 15(4), 101-115.
- **Hamilton, J. D.** (1994). Time series analysis. Princeton university press.
- **Lütkepohl, H.** (2005). New introduction to multiple time series analysis. Springer.
- **Sims, C. A.** (1980). Macroeconomics and reality. Econometrica, 1-48.

## 🎉 Enjoy Exploring Macroeconomic Dynamics!

This dashboard puts sophisticated econometric tools in your hands. Have fun exploring how economies respond to different shocks, comparing dynamics across countries, and building your intuition about macroeconomic relationships.

Happy analyzing! 📊🌍
