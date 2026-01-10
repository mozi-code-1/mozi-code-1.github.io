"""
Interactive Macroeconomic VAR Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta

from data_fetcher import DataFetcher, get_data_summary, align_date_ranges
from var_models import (
    StationarityTester,
    CointegrationTester,
    VAREstimator,
    VECMEstimator,
    RobustVAREstimator
)

# Page configuration
st.set_page_config(
    page_title="Macroeconomic VAR Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_data(country, start_date, end_date, api_key, covid_handling="No adjustment"):
    """Cached data fetching"""
    fetcher = DataFetcher(api_key)
    return fetcher.fetch_country_data(country, start_date, end_date, covid_handling)


def plot_time_series(data: pd.DataFrame, title: str = "Time Series"):
    """Plot time series data"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=data.columns.tolist(),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, col in enumerate(data.columns):
        row = (i // 2) + 1
        col_num = (i % 2) + 1

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data[col],
                mode='lines',
                name=col,
                line=dict(color=colors[i], width=2),
                showlegend=False
            ),
            row=row, col=col_num
        )

        fig.update_xaxes(title_text="Date", row=row, col=col_num)
        fig.update_yaxes(title_text=col, row=row, col=col_num)

    fig.update_layout(
        height=600,
        title_text=title,
        title_font_size=16,
        showlegend=False
    )

    return fig


def plot_irf(irfs, lower, upper, var_names, shock_var, response_var, periods=24):
    """Plot impulse response function with confidence bands"""
    shock_idx = var_names.index(shock_var)
    response_idx = var_names.index(response_var)

    # Extract relevant IRF
    irf_values = irfs[:, response_idx, shock_idx]
    lower_values = lower[:, response_idx, shock_idx]
    upper_values = upper[:, response_idx, shock_idx]

    periods_range = list(range(periods))

    fig = go.Figure()

    # Add confidence bands
    fig.add_trace(go.Scatter(
        x=periods_range + periods_range[::-1],
        y=upper_values.tolist() + lower_values.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(31, 119, 180, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='90% CI'
    ))

    # Add IRF line
    fig.add_trace(go.Scatter(
        x=periods_range,
        y=irf_values,
        mode='lines',
        name='IRF',
        line=dict(color='#1f77b4', width=2)
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=f"Response of {response_var} to {shock_var} Shock",
        xaxis_title="Periods (Quarters)",
        yaxis_title="Response",
        height=400,
        hovermode='x unified'
    )

    return fig


def plot_all_irfs(irfs, var_names, periods=24):
    """Plot grid of all IRFs"""
    n_vars = len(var_names)

    fig = make_subplots(
        rows=n_vars, cols=n_vars,
        subplot_titles=[f"{r} ← {c}" for c in var_names for r in var_names],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    for i in range(n_vars):  # Response
        for j in range(n_vars):  # Shock
            irf_values = irfs[:, i, j]

            fig.add_trace(
                go.Scatter(
                    x=list(range(periods)),
                    y=irf_values,
                    mode='lines',
                    line=dict(color='#1f77b4', width=1.5),
                    showlegend=False
                ),
                row=i+1, col=j+1
            )

            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                opacity=0.3,
                row=i+1, col=j+1
            )

    fig.update_layout(
        height=800,
        title_text="All Impulse Response Functions",
        showlegend=False
    )

    return fig


def plot_forecast(historical, forecast, lower, upper, variable):
    """Plot forecast with confidence intervals"""
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical[variable],
        mode='lines',
        name='Historical',
        line=dict(color='#1f77b4', width=2)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast[variable],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast.index.tolist() + forecast.index.tolist()[::-1],
        y=upper[variable].tolist() + lower[variable].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 127, 14, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='95% CI'
    ))

    fig.update_layout(
        title=f"Forecast: {variable}",
        xaxis_title="Date",
        yaxis_title=variable,
        height=400,
        hovermode='x unified'
    )

    return fig


def plot_fevd(fevd_df, response_var, periods=24):
    """Plot forecast error variance decomposition"""
    # Filter for specific response variable
    fevd_var = fevd_df[fevd_df['Response'] == response_var].copy()

    # Get shock columns
    shock_cols = [col for col in fevd_var.columns if col.startswith('Shock:')]

    fig = go.Figure()

    # Stack area plot
    for col in shock_cols:
        fig.add_trace(go.Scatter(
            x=fevd_var['Step'],
            y=fevd_var[col] * 100,  # Convert to percentage
            mode='lines',
            name=col.replace('Shock: ', ''),
            stackgroup='one',
            hovertemplate='%{y:.1f}%'
        ))

    fig.update_layout(
        title=f"Forecast Error Variance Decomposition: {response_var}",
        xaxis_title="Quarters Ahead",
        yaxis_title="Percentage",
        height=400,
        hovermode='x unified',
        yaxis=dict(ticksuffix='%')
    )

    return fig


def plot_historical_decomposition(decomp_df, variable):
    """Plot historical decomposition"""
    shock_cols = [col for col in decomp_df.columns if col.startswith('Shock:')]

    fig = go.Figure()

    # Add each shock's contribution
    for col in shock_cols:
        fig.add_trace(go.Scatter(
            x=decomp_df.index,
            y=decomp_df[col],
            mode='lines',
            name=col.replace('Shock: ', ''),
            stackgroup='one'
        ))

    # Add actual values
    fig.add_trace(go.Scatter(
        x=decomp_df.index,
        y=decomp_df['Actual'],
        mode='lines',
        name='Actual',
        line=dict(color='black', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"Historical Decomposition: {variable}",
        xaxis_title="Date",
        yaxis_title=variable,
        height=500,
        hovermode='x unified'
    )

    return fig


def main():
    """Main application"""

    # Header
    st.markdown('<p class="main-header">📊 Macroeconomic VAR Dashboard</p>', unsafe_allow_html=True)
    st.markdown("Compare macroeconomic dynamics across countries using Vector Autoregression (VAR) models")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # API Key input
    api_key = st.sidebar.text_input(
        "FRED API Key",
        type="password",
        help="Get your free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )

    if not api_key:
        st.warning("⚠️ Please enter your FRED API key in the sidebar to continue.")
        st.info("""
        **How to get a FRED API Key:**
        1. Go to https://fred.stlouisfed.org/
        2. Create a free account
        3. Request an API key at https://fred.stlouisfed.org/docs/api/api_key.html
        4. Copy and paste the key in the sidebar
        """)
        return

    # Initialize data fetcher
    fetcher = DataFetcher(api_key)
    available_countries = fetcher.get_available_countries()

    # Country selection
    st.sidebar.subheader("🌍 Country Selection")
    country1 = st.sidebar.selectbox("Country 1", available_countries, index=0)
    country2 = st.sidebar.selectbox("Country 2", available_countries, index=1)

    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    col1, col2 = st.sidebar.columns(2)

    default_start = datetime(2000, 1, 1)
    default_end = datetime.now()

    start_date = col1.date_input(
        "Start Date",
        value=default_start,
        min_value=datetime(1960, 1, 1),
        max_value=default_end
    )

    end_date = col2.date_input(
        "End Date",
        value=default_end,
        min_value=start_date,
        max_value=datetime.now()
    )

    # Model configuration
    st.sidebar.subheader("🔧 Model Settings")

    max_lags = st.sidebar.slider("Maximum Lags to Consider", 1, 12, 8)
    ic_choice = st.sidebar.selectbox("Information Criterion", ['aic', 'bic', 'hqic'], index=1)

    identification = st.sidebar.radio(
        "Identification Scheme",
        ["Cholesky Decomposition", "Blanchard-Quah Long-Run Restrictions"]
    )

    irf_periods = st.sidebar.slider("IRF Periods", 4, 40, 24)
    forecast_periods = st.sidebar.slider("Forecast Periods", 4, 20, 12)

    # COVID-19 handling options
    st.sidebar.subheader("🦠 COVID-19 Handling")
    covid_handling = st.sidebar.radio(
        "How to handle COVID-19 period?",
        [
            "No adjustment",
            "Exclude COVID period",
            "COVID dummy variable",
            "Fat-tailed errors (Student's t)"
        ],
        help="""
        - **No adjustment**: Use raw data including COVID period
        - **Exclude COVID period**: Remove 2020Q1-2021Q4 entirely
        - **COVID dummy variable**: Include dummy for COVID quarters as exogenous variable
        - **Fat-tailed errors**: Use Student's t errors to downweight outliers (ECB recommended)
        """
    )

    # Fetch data button
    if st.sidebar.button("🔄 Fetch Data and Estimate Models", type="primary"):
        with st.spinner(f"Fetching data for {country1} and {country2}..."):
            try:
                # Fetch data for both countries with COVID handling
                data1, info1 = fetch_data(country1, str(start_date), str(end_date), api_key, covid_handling)
                data2, info2 = fetch_data(country2, str(start_date), str(end_date), api_key, covid_handling)

                if len(data1) < 20 or len(data2) < 20:
                    st.error("Insufficient data. Please select a different date range or countries.")
                    return

                # Align date ranges between countries
                data1_aligned, data2_aligned = align_date_ranges(data1, data2)

                if len(data1_aligned) < 20 or len(data2_aligned) < 20:
                    st.error("Insufficient overlapping data. Please select a different date range or countries.")
                    return

                # Store in session state
                st.session_state['data1'] = data1_aligned
                st.session_state['data2'] = data2_aligned
                st.session_state['country1'] = country1
                st.session_state['country2'] = country2
                st.session_state['info1'] = info1
                st.session_state['info2'] = info2
                st.session_state['covid_handling'] = covid_handling

                st.success(f"✅ Data fetched successfully! Date range: {data1_aligned.index.min().strftime('%Y-%m-%d')} to {data1_aligned.index.max().strftime('%Y-%m-%d')}")
                st.info(f"COVID-19 handling: {covid_handling}")

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                return

    # Check if data is available
    if 'data1' not in st.session_state:
        st.info("👈 Configure settings in the sidebar and click 'Fetch Data and Estimate Models' to begin.")
        return

    # Get data from session state
    data1 = st.session_state['data1']
    data2 = st.session_state['data2']
    country1 = st.session_state['country1']
    country2 = st.session_state['country2']

    # Create tabs for different sections
    tabs = st.tabs([
        "📈 Data Overview",
        "🔍 Stationarity & Cointegration",
        "📊 VAR/VECM Estimation",
        "🩺 Model Diagnostics",
        "💥 Impulse Responses",
        "🔮 Forecasting",
        "🔗 Granger Causality",
        "📉 Historical Decomposition"
    ])

    # TAB 1: Data Overview
    with tabs[0]:
        st.markdown('<p class="section-header">Raw Time Series Data</p>', unsafe_allow_html=True)

        # Filter out COVID dummy for visualization
        data1_plot = data1.drop('COVID_DUMMY', axis=1) if 'COVID_DUMMY' in data1.columns else data1
        data2_plot = data2.drop('COVID_DUMMY', axis=1) if 'COVID_DUMMY' in data2.columns else data2

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{country1}")
            st.plotly_chart(
                plot_time_series(data1_plot, f"{country1} - Macroeconomic Variables"),
                use_container_width=True
            )

            with st.expander("📊 Summary Statistics"):
                st.dataframe(get_data_summary(data1_plot), use_container_width=True)

            with st.expander("📋 Data Sample"):
                st.dataframe(data1.tail(10), use_container_width=True)

        with col2:
            st.subheader(f"{country2}")
            st.plotly_chart(
                plot_time_series(data2_plot, f"{country2} - Macroeconomic Variables"),
                use_container_width=True
            )

            with st.expander("📊 Summary Statistics"):
                st.dataframe(get_data_summary(data2_plot), use_container_width=True)

            with st.expander("📋 Data Sample"):
                st.dataframe(data2.tail(10), use_container_width=True)

    # TAB 2: Stationarity & Cointegration
    with tabs[1]:
        st.markdown('<p class="section-header">Stationarity Tests (ADF)</p>', unsafe_allow_html=True)

        # Econometric guidance
        with st.expander("📚 Stationarity & Model Selection Guidance"):
            st.markdown("""
            **How to choose between VAR and VECM:**

            1. **All variables stationary (ADF p-value < 0.05):**
               - ✅ Use VAR in levels
               - Variables return to their mean after shocks

            2. **Unit roots detected BUT no cointegration:**
               - ✅ Use VAR in first differences
               - Variables drift apart permanently
               - Note: Differencing loses long-run information

            3. **Unit roots detected AND cointegration found:**
               - ✅ Use VECM (Vector Error Correction Model)
               - Variables share long-run equilibrium relationship
               - VECM preserves both short-run dynamics and long-run equilibrium

            **What is Cointegration?**
            - Non-stationary variables that share a common stochastic trend
            - They may drift apart temporarily but return to long-run relationship
            - Example: GDP and consumption often cointegrate (permanent income hypothesis)

            **Johansen Test interpretation:**
            - Tests for number of cointegrating relationships (rank)
            - If rank > 0: Use VECM with that many cointegrating vectors
            - If rank = 0: No cointegration, use VAR in differences
            """)

        # Filter out COVID dummy for tests
        data1_test = data1.drop('COVID_DUMMY', axis=1) if 'COVID_DUMMY' in data1.columns else data1
        data2_test = data2.drop('COVID_DUMMY', axis=1) if 'COVID_DUMMY' in data2.columns else data2

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{country1}")
            adf_results1 = StationarityTester.test_dataframe(data1_test)
            st.dataframe(adf_results1, use_container_width=True)

        with col2:
            st.subheader(f"{country2}")
            adf_results2 = StationarityTester.test_dataframe(data2_test)
            st.dataframe(adf_results2, use_container_width=True)

        st.markdown('<p class="section-header">Johansen Cointegration Test</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{country1}")
            try:
                coint1 = CointegrationTester.johansen_test(data1_test, det_order=0, k_ar_diff=2)

                st.write("**Trace Statistic Test:**")
                st.dataframe(coint1['trace_results'], use_container_width=True)

                st.metric("Cointegration Rank", coint1['cointegration_rank'])

                if coint1['has_cointegration']:
                    st.success(f"✅ Cointegration detected (rank = {coint1['cointegration_rank']})")
                    st.info("💡 A VECM should be estimated instead of VAR in levels")
                else:
                    st.info("ℹ️ No cointegration detected. VAR in differences may be appropriate.")

                st.session_state['coint1'] = coint1

            except Exception as e:
                st.error(f"Error in cointegration test: {str(e)}")

        with col2:
            st.subheader(f"{country2}")
            try:
                coint2 = CointegrationTester.johansen_test(data2_test, det_order=0, k_ar_diff=2)

                st.write("**Trace Statistic Test:**")
                st.dataframe(coint2['trace_results'], use_container_width=True)

                st.metric("Cointegration Rank", coint2['cointegration_rank'])

                if coint2['has_cointegration']:
                    st.success(f"✅ Cointegration detected (rank = {coint2['cointegration_rank']})")
                    st.info("💡 A VECM should be estimated instead of VAR in levels")
                else:
                    st.info("ℹ️ No cointegration detected. VAR in differences may be appropriate.")

                st.session_state['coint2'] = coint2

            except Exception as e:
                st.error(f"Error in cointegration test: {str(e)}")

    # TAB 3: VAR/VECM Estimation
    with tabs[2]:
        st.markdown('<p class="section-header">Model Estimation</p>', unsafe_allow_html=True)

        # Econometric best practices warnings
        st.info("⚠️ **Variable Ordering Matters**: For Cholesky identification, variable ordering determines which variables respond contemporaneously. " +
                "Common macro ordering: slow-moving variables (GDP, unemployment) before fast-moving variables (inflation, interest rates).")

        # Variable ordering controls
        with st.expander("🔧 Reorder Variables"):
            st.write("Current order: " + ", ".join(data1.drop('COVID_DUMMY', axis=1, errors='ignore').columns if 'COVID_DUMMY' in data1.columns else data1.columns))
            st.write("**Recommended ordering:** GDP growth → Unemployment → Inflation → Interest rate")
            if st.button("Apply Recommended Ordering"):
                # Define recommended order
                recommended_order = []
                current_cols = data1.drop('COVID_DUMMY', axis=1, errors='ignore').columns if 'COVID_DUMMY' in data1.columns else data1.columns

                for var_type in ['gdp_growth', 'unemployment', 'inflation', 'interest_rate']:
                    for col in current_cols:
                        if var_type in col.lower():
                            recommended_order.append(col)

                if len(recommended_order) == len(current_cols):
                    data1 = data1[recommended_order + (['COVID_DUMMY'] if 'COVID_DUMMY' in data1.columns else [])]
                    data2 = data2[recommended_order + (['COVID_DUMMY'] if 'COVID_DUMMY' in data2.columns else [])]
                    st.success("✅ Variables reordered!")
                    st.rerun()

        col1, col2 = st.columns(2)

        # Estimate VAR for Country 1
        with col1:
            st.subheader(f"{country1}")

            with st.spinner("Estimating model..."):
                try:
                    # Check if should use VECM
                    use_vecm1 = ('coint1' in st.session_state and
                                st.session_state['coint1']['has_cointegration'])

                    if use_vecm1:
                        st.info("📊 Estimating VECM (cointegration detected)")
                        coint_rank = st.session_state['coint1']['cointegration_rank']

                        vecm1 = VECMEstimator(data1, coint_rank=coint_rank)
                        vecm1.fit(k_ar_diff=2)

                        st.session_state['model1'] = vecm1
                        st.session_state['model_type1'] = 'VECM'

                        with st.expander("📋 Model Summary"):
                            st.text(vecm1.get_summary())

                        with st.expander("🔗 Cointegrating Vectors"):
                            st.dataframe(vecm1.get_cointegration_vectors(), use_container_width=True)

                    else:
                        # Get COVID handling setting
                        covid_handling = st.session_state.get('covid_handling', 'No adjustment')

                        # Extract exogenous variables if COVID dummy is used
                        exog1 = None
                        data1_endog = data1
                        if 'COVID_DUMMY' in data1.columns:
                            exog1 = data1[['COVID_DUMMY']]
                            data1_endog = data1.drop('COVID_DUMMY', axis=1)
                            st.info("📊 Estimating VAR model with COVID dummy as exogenous variable")
                        elif covid_handling == "Fat-tailed errors (Student's t)":
                            st.info("📊 Estimating Robust VAR model with Student's t errors")
                        else:
                            st.info("📊 Estimating VAR model")

                        # Choose estimator based on COVID handling
                        if covid_handling == "Fat-tailed errors (Student's t)":
                            var1 = RobustVAREstimator(data1_endog, exog=exog1, df_t=5.0)
                        else:
                            var1 = VAREstimator(data1_endog, exog=exog1)

                        # Lag selection
                        lag_selection = var1.select_lag_order(maxlags=max_lags)

                        st.write("**Lag Order Selection:**")
                        st.write("📊 BIC is preferred for VAR models (penalizes complexity more)")

                        # Enhanced lag selection display with all criteria
                        with st.expander("📈 Detailed Lag Selection Criteria"):
                            for lag in range(1, min(max_lags + 1, 9)):
                                if lag <= max_lags:
                                    lag_info = lag_selection.get(lag, {})
                                    if lag_info:
                                        st.write(f"**Lag {lag}:**")
                                        cols = st.columns(4)
                                        cols[0].metric("AIC", f"{lag_info.get('aic', 0):.2f}")
                                        cols[1].metric("BIC", f"{lag_info.get('bic', 0):.2f}")
                                        cols[2].metric("HQ", f"{lag_info.get('hqic', 0):.2f}")
                                        cols[3].metric("FPE", f"{lag_info.get('fpe', 0):.4f}")

                        lag_df = pd.DataFrame({
                            'Criterion': ['AIC', 'BIC', 'HQIC'],
                            'Selected Lags': [
                                lag_selection['selected_aic'],
                                lag_selection['selected_bic'],
                                lag_selection['selected_hqic']
                            ]
                        })
                        st.dataframe(lag_df, use_container_width=True)

                        # Fit VAR
                        var1.fit(ic=ic_choice)

                        st.metric("Selected Lag Order", var1.lag_order)

                        # Overfitting warning
                        n_vars = len(data1_endog.columns)
                        n_params = n_vars**2 * var1.lag_order + n_vars  # k^2 * p + k
                        n_obs = len(data1_endog) - var1.lag_order
                        param_ratio = n_params / n_obs

                        if param_ratio > 0.1:
                            st.warning(f"⚠️ **Overfitting Risk**: {n_params} parameters with {n_obs} observations " +
                                     f"(ratio: {param_ratio:.2%}). Consider reducing lags or variables.")

                        st.session_state['model1'] = var1
                        st.session_state['model_type1'] = 'VAR'

                        with st.expander("📋 Model Summary"):
                            st.text(var1.get_summary())

                        # Diagnostics
                        diagnostics1 = var1.get_diagnostics()

                        if diagnostics1:
                            with st.expander("🔍 Diagnostic Tests"):
                                for test_name, test_result in diagnostics1.items():
                                    st.write(f"**{test_name}:**")
                                    st.write(f"- Statistic: {test_result.get('statistic', 'N/A'):.4f}")
                                    st.write(f"- p-value: {test_result.get('p_value', 'N/A'):.4f}")
                                    st.write(f"- Null: {test_result.get('null_hypothesis', 'N/A')}")

                except Exception as e:
                    st.error(f"Error estimating model: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

        # Estimate VAR for Country 2
        with col2:
            st.subheader(f"{country2}")

            with st.spinner("Estimating model..."):
                try:
                    # Check if should use VECM
                    use_vecm2 = ('coint2' in st.session_state and
                                st.session_state['coint2']['has_cointegration'])

                    if use_vecm2:
                        st.info("📊 Estimating VECM (cointegration detected)")
                        coint_rank = st.session_state['coint2']['cointegration_rank']

                        vecm2 = VECMEstimator(data2, coint_rank=coint_rank)
                        vecm2.fit(k_ar_diff=2)

                        st.session_state['model2'] = vecm2
                        st.session_state['model_type2'] = 'VECM'

                        with st.expander("📋 Model Summary"):
                            st.text(vecm2.get_summary())

                        with st.expander("🔗 Cointegrating Vectors"):
                            st.dataframe(vecm2.get_cointegration_vectors(), use_container_width=True)

                    else:
                        # Get COVID handling setting
                        covid_handling = st.session_state.get('covid_handling', 'No adjustment')

                        # Extract exogenous variables if COVID dummy is used
                        exog2 = None
                        data2_endog = data2
                        if 'COVID_DUMMY' in data2.columns:
                            exog2 = data2[['COVID_DUMMY']]
                            data2_endog = data2.drop('COVID_DUMMY', axis=1)
                            st.info("📊 Estimating VAR model with COVID dummy as exogenous variable")
                        elif covid_handling == "Fat-tailed errors (Student's t)":
                            st.info("📊 Estimating Robust VAR model with Student's t errors")
                        else:
                            st.info("📊 Estimating VAR model")

                        # Choose estimator based on COVID handling
                        if covid_handling == "Fat-tailed errors (Student's t)":
                            var2 = RobustVAREstimator(data2_endog, exog=exog2, df_t=5.0)
                        else:
                            var2 = VAREstimator(data2_endog, exog=exog2)

                        # Lag selection
                        lag_selection = var2.select_lag_order(maxlags=max_lags)

                        st.write("**Lag Order Selection:**")
                        lag_df = pd.DataFrame({
                            'Criterion': ['AIC', 'BIC', 'HQIC'],
                            'Selected Lags': [
                                lag_selection['selected_aic'],
                                lag_selection['selected_bic'],
                                lag_selection['selected_hqic']
                            ]
                        })
                        st.dataframe(lag_df, use_container_width=True)

                        # Fit VAR
                        var2.fit(ic=ic_choice)

                        st.metric("Selected Lag Order", var2.lag_order)

                        st.session_state['model2'] = var2
                        st.session_state['model_type2'] = 'VAR'

                        with st.expander("📋 Model Summary"):
                            st.text(var2.get_summary())

                        # Diagnostics
                        diagnostics2 = var2.get_diagnostics()

                        if diagnostics2:
                            with st.expander("🔍 Diagnostic Tests"):
                                for test_name, test_result in diagnostics2.items():
                                    st.write(f"**{test_name}:**")
                                    st.write(f"- Statistic: {test_result.get('statistic', 'N/A'):.4f}")
                                    st.write(f"- p-value: {test_result.get('p_value', 'N/A'):.4f}")
                                    st.write(f"- Null: {test_result.get('null_hypothesis', 'N/A')}")

                except Exception as e:
                    st.error(f"Error estimating model: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # TAB 4: Model Diagnostics
    with tabs[3]:
        st.markdown('<p class="section-header">Model Diagnostics & Tests</p>', unsafe_allow_html=True)

        if 'model1' not in st.session_state or 'model2' not in st.session_state:
            st.warning("⚠️ Please estimate models first (go to VAR/VECM Estimation tab)")
        else:
            country_choice = st.selectbox("Select Country for Diagnostics", [country1, country2], key='diag_country')

            model = st.session_state['model1'] if country_choice == country1 else st.session_state['model2']
            model_type = st.session_state['model_type1'] if country_choice == country1 else st.session_state['model_type2']

            # Both VAR and VECM now support diagnostics
            if model_type == 'VAR':
                st.info(f"📊 Showing diagnostics for {model_type} model with {model.lag_order} lags")
            else:
                st.info(f"📊 Showing diagnostics for {model_type} model with cointegration rank {model.coint_rank}")

            diagnostics = model.get_diagnostics()

            # VAR-specific diagnostics
            if model_type == 'VAR':

                # Summary section
                st.markdown("### 🎯 Overall Model Quality")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    passes_autocorr = diagnostics.get('autocorrelation', {}).get('passes', False)
                    st.metric("Autocorrelation", "✅ Pass" if passes_autocorr else "❌ Fail")

                with col2:
                    passes_normal = diagnostics.get('normality', {}).get('passes', False)
                    st.metric("Normality", "✅ Pass" if passes_normal else "❌ Fail")

                with col3:
                    passes_het = diagnostics.get('heteroscedasticity', {}).get('passes', False)
                    st.metric("Homoscedasticity", "✅ Pass" if passes_het else "❌ Fail")

                with col4:
                    passes_stab = diagnostics.get('stability', {}).get('passes', False)
                    st.metric("Stability", "✅ Pass" if passes_stab else "❌ Fail")

                # 1. Residual Autocorrelation Test
                st.markdown("### 1️⃣ Residual Autocorrelation (Portmanteau/LM Test)")
                if 'autocorrelation' in diagnostics and 'error' not in diagnostics['autocorrelation']:
                    auto_test = diagnostics['autocorrelation']
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Test Statistic", f"{auto_test['statistic']:.4f}")
                        st.metric("p-value", f"{auto_test['p_value']:.4f}")

                    with col2:
                        if auto_test['passes']:
                            st.success("✅ " + auto_test['interpretation'])
                        else:
                            st.error("❌ " + auto_test['interpretation'])

                    st.caption("**Null Hypothesis:** No residual autocorrelation at any lag")
                    st.caption("**What it means:** Tests if residuals are serially correlated. If fails, model may be misspecified.")

                # 2. Normality Test
                st.markdown("### 2️⃣ Normality Test (Jarque-Bera)")
                if 'normality' in diagnostics and 'error' not in diagnostics['normality']:
                    norm_test = diagnostics['normality']
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Test Statistic", f"{norm_test['statistic']:.4f}")
                        st.metric("p-value", f"{norm_test['p_value']:.4f}")

                    with col2:
                        if norm_test['passes']:
                            st.success("✅ " + norm_test['interpretation'])
                        else:
                            st.warning("⚠️ " + norm_test['interpretation'])

                    st.caption("**Null Hypothesis:** Residuals are normally distributed")
                    st.caption("**What it means:** Tests if residuals follow a normal distribution. Failure may indicate outliers or need for robust estimation.")

                # 3. Heteroscedasticity Test
                st.markdown("### 3️⃣ Heteroscedasticity Test (White's Test)")
                if 'heteroscedasticity' in diagnostics and 'error' not in diagnostics['heteroscedasticity']:
                    het_test = diagnostics['heteroscedasticity']

                    if 'by_equation' in het_test:
                        het_df = pd.DataFrame(het_test['by_equation'])
                        st.dataframe(het_df, use_container_width=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average p-value", f"{het_test.get('average_p_value', 0):.4f}")

                    with col2:
                        if het_test['passes']:
                            st.success("✅ " + het_test['interpretation'])
                        else:
                            st.warning("⚠️ " + het_test['interpretation'])

                    st.caption("**Null Hypothesis:** Homoscedastic residuals (constant variance)")
                    st.caption("**What it means:** Tests if error variance changes over time. Failure suggests using robust standard errors.")

                # 4. Stability Check
                st.markdown("### 4️⃣ Stability Check (AR Characteristic Roots)")
                if 'stability' in diagnostics and 'error' not in diagnostics['stability']:
                    stab_test = diagnostics['stability']

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Max Root Modulus", f"{stab_test['max_root_modulus']:.4f}")
                        st.metric("Number of Roots", len(stab_test['all_roots']))

                    with col2:
                        if stab_test['passes']:
                            st.success("✅ " + stab_test['interpretation'])
                        else:
                            st.error("❌ " + stab_test['interpretation'])

                    # Plot roots
                    fig = go.Figure()

                    # Unit circle
                    theta = np.linspace(0, 2*np.pi, 100)
                    fig.add_trace(go.Scatter(
                        x=np.cos(theta),
                        y=np.sin(theta),
                        mode='lines',
                        name='Unit Circle',
                        line=dict(color='red', dash='dash')
                    ))

                    # Roots
                    roots_array = np.array([complex(r) if isinstance(r, (int, float)) else r for r in stab_test['all_roots']])
                    fig.add_trace(go.Scatter(
                        x=[0],
                        y=[0],
                        mode='markers',
                        marker=dict(size=10, color='blue'),
                        name='AR Roots',
                        text=[f"Root {i+1}: {abs(r):.3f}" for i, r in enumerate(roots_array)],
                        hoverinfo='text'
                    ))

                    fig.update_layout(
                        title="AR Characteristic Roots",
                        xaxis_title="Real Part",
                        yaxis_title="Imaginary Part",
                        height=400,
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("**What it means:** All roots must be inside the unit circle for the model to be stable. Unstable models produce unreliable forecasts.")

                # 5. Model Summary Statistics
                st.markdown("### 5️⃣ Model Summary Statistics")
                if 'model_stats' in diagnostics and 'error' not in diagnostics['model_stats']:
                    stats = diagnostics['model_stats']

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("AIC", f"{stats.get('aic', 0):.2f}")
                    with col2:
                        st.metric("BIC", f"{stats.get('bic', 0):.2f}")
                    with col3:
                        st.metric("HQIC", f"{stats.get('hqic', 0):.2f}")
                    with col4:
                        if stats.get('log_likelihood'):
                            st.metric("Log-Likelihood", f"{stats['log_likelihood']:.2f}")

                    st.markdown("**R-squared by Equation:**")
                    if 'by_equation' in stats:
                        stats_df = pd.DataFrame(stats['by_equation'])
                        st.dataframe(stats_df, use_container_width=True)

                    st.caption("**What it means:** Lower AIC/BIC indicates better model fit. R² shows proportion of variance explained.")

                # 6. Residual Statistics
                st.markdown("### 6️⃣ Residual Statistics")
                if 'residual_stats' in diagnostics and 'error' not in diagnostics['residual_stats']:
                    res_stats = diagnostics['residual_stats']

                    resid_df = pd.DataFrame({
                        'Mean': res_stats.get('means', {}),
                        'Std Dev': res_stats.get('std_devs', {}),
                        'Skewness': res_stats.get('skewness', {}),
                        'Kurtosis': res_stats.get('kurtosis', {})
                    })

                    st.dataframe(resid_df, use_container_width=True)
                    st.caption("**What it means:** Means should be near 0. Skewness and kurtosis indicate departure from normality.")

                # 7. Residual Plots
                st.markdown("### 7️⃣ Residual Plots")
                plot_data = model.get_residual_plots_data()

                var_select = st.selectbox("Select Variable for Residual Plots", list(plot_data.keys()))

                if var_select and var_select in plot_data and 'error' not in plot_data[var_select]:
                    var_data = plot_data[var_select]

                    # Create subplots
                    fig = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=["Residuals Over Time", "ACF", "PACF"],
                        vertical_spacing=0.1
                    )

                    # Residuals over time
                    fig.add_trace(go.Scatter(
                        x=var_data['dates'],
                        y=var_data['residuals'],
                        mode='lines',
                        name='Residuals',
                        line=dict(color='blue')
                    ), row=1, col=1)

                    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

                    # ACF
                    lags_acf = list(range(len(var_data['acf'])))
                    fig.add_trace(go.Bar(
                        x=lags_acf,
                        y=var_data['acf'],
                        name='ACF',
                        marker_color='green'
                    ), row=2, col=1)

                    # Add confidence bands for ACF
                    conf_level = 1.96 / np.sqrt(len(var_data['residuals']))
                    fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=2, col=1)

                    # PACF
                    lags_pacf = list(range(len(var_data['pacf'])))
                    fig.add_trace(go.Bar(
                        x=lags_pacf,
                        y=var_data['pacf'],
                        name='PACF',
                        marker_color='orange'
                    ), row=3, col=1)

                    # Add confidence bands for PACF
                    fig.add_hline(y=conf_level, line_dash="dash", line_color="red", row=3, col=1)
                    fig.add_hline(y=-conf_level, line_dash="dash", line_color="red", row=3, col=1)

                    fig.update_layout(height=900, showlegend=False)
                    fig.update_xaxes(title_text="Date", row=1, col=1)
                    fig.update_xaxes(title_text="Lag", row=2, col=1)
                    fig.update_xaxes(title_text="Lag", row=3, col=1)

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption("**What it means:** Residuals should be random (no patterns). ACF/PACF should be mostly within confidence bands (red dashed lines).")

            else:  # VECM diagnostics
                # Summary section
                st.markdown("### 🎯 Overall Model Quality (VECM)")
                col1, col2, col3 = st.columns(3)

                with col1:
                    passes_autocorr = diagnostics.get('autocorrelation', {}).get('passes', False)
                    st.metric("Autocorrelation", "✅ Pass" if passes_autocorr else "❌ Fail")

                with col2:
                    passes_normal = diagnostics.get('normality', {}).get('passes', False)
                    st.metric("Normality", "✅ Pass" if passes_normal else "❌ Fail")

                with col3:
                    coint_rank = diagnostics.get('cointegration', {}).get('rank', 0)
                    st.metric("Coint. Rank", coint_rank)

                # 1. Autocorrelation Test
                st.markdown("### 1️⃣ Residual Autocorrelation")
                if 'autocorrelation' in diagnostics and 'error' not in diagnostics['autocorrelation']:
                    auto_test = diagnostics['autocorrelation']
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Test Statistic", f"{auto_test['statistic']:.4f}")
                        st.metric("p-value", f"{auto_test['p_value']:.4f}")

                    with col2:
                        if auto_test['passes']:
                            st.success("✅ " + auto_test['interpretation'])
                        else:
                            st.error("❌ " + auto_test['interpretation'])

                # 2. Normality Test
                st.markdown("### 2️⃣ Normality Test")
                if 'normality' in diagnostics and 'error' not in diagnostics['normality']:
                    norm_test = diagnostics['normality']
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Test Statistic", f"{norm_test['statistic']:.4f}")
                        st.metric("p-value", f"{norm_test['p_value']:.4f}")

                    with col2:
                        if norm_test['passes']:
                            st.success("✅ " + norm_test['interpretation'])
                        else:
                            st.warning("⚠️ " + norm_test['interpretation'])

                # 3. Model Information Criteria
                st.markdown("### 3️⃣ Model Information Criteria")
                if 'model_stats' in diagnostics and 'error' not in diagnostics['model_stats']:
                    stats = diagnostics['model_stats']
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("AIC", f"{stats.get('aic', 0):.2f}")

                    with col2:
                        st.metric("BIC", f"{stats.get('bic', 0):.2f}")

                    with col3:
                        st.metric("HQIC", f"{stats.get('hqic', 0):.2f}")

                    st.caption("**Lower values are better.** BIC penalizes complexity most.")

                # 4. Cointegration Information
                st.markdown("### 4️⃣ Cointegration Diagnostics")
                if 'cointegration' in diagnostics:
                    coint_info = diagnostics['cointegration']
                    st.info(f"📊 {coint_info.get('interpretation', 'N/A')}")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Cointegration Rank", coint_info.get('rank', 0))
                    with col2:
                        st.metric("Number of Variables", coint_info.get('num_variables', 0))
                    st.caption("**What it means:** VECM models long-run equilibrium relationships between non-stationary variables.")

    # TAB 5: Impulse Response Functions
    with tabs[4]:
        st.markdown('<p class="section-header">Impulse Response Functions</p>', unsafe_allow_html=True)

        if 'model1' not in st.session_state or 'model2' not in st.session_state:
            st.warning("⚠️ Please estimate models first (go to VAR/VECM Estimation tab)")
        else:
            # IRF settings
            col1, col2, col3 = st.columns(3)

            with col1:
                country_choice = st.selectbox("Select Country", [country1, country2])

            with col2:
                var_names = list(data1.columns)
                shock_var = st.selectbox("Shock Variable", var_names)

            with col3:
                response_var = st.selectbox("Response Variable", var_names)

            confidence_level = st.select_slider(
                "Confidence Level",
                options=[68, 90, 95],
                value=90
            )

            # Calculate IRFs
            model = st.session_state['model1'] if country_choice == country1 else st.session_state['model2']
            model_type = st.session_state['model_type1'] if country_choice == country1 else st.session_state['model_type2']

            # Both VAR and VECM support IRFs
            with st.spinner(f"Computing impulse responses using {model_type} model..."):
                try:
                    alpha = (100 - confidence_level) / 100

                    id_scheme = 'cholesky' if 'Cholesky' in identification else 'long_run'

                    if model_type == 'VECM':
                        st.info("💡 VECM IRFs show long-run effects including cointegration adjustments")
                        # VECM only supports Cholesky for now
                        id_scheme = 'cholesky'

                    irfs, lower, upper = model.irf_with_confidence_bands(
                        periods=irf_periods,
                        alpha=alpha,
                        identification=id_scheme,
                        n_bootstrap=500 if model_type == 'VAR' else 200  # Fewer for VECM (slower)
                    )

                    # Plot specific IRF
                    st.plotly_chart(
                        plot_irf(irfs, lower, upper, var_names, shock_var, response_var, irf_periods),
                        use_container_width=True
                    )

                    # Show all IRFs
                    with st.expander("📊 View All IRFs"):
                        st.plotly_chart(
                            plot_all_irfs(irfs, var_names, irf_periods),
                            use_container_width=True
                        )

                    # Download IRF data
                    irf_data = pd.DataFrame(
                        irfs[:, var_names.index(response_var), var_names.index(shock_var)],
                        columns=[f'{response_var} response to {shock_var}']
                    )

                    csv = irf_data.to_csv(index=True)
                    st.download_button(
                        label="📥 Download IRF Data (CSV)",
                        data=csv,
                        file_name=f"irf_{country_choice}_{model_type}_{shock_var}_{response_var}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error computing IRFs: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # TAB 6: Forecasting
    with tabs[5]:
        st.markdown('<p class="section-header">Forecasting</p>', unsafe_allow_html=True)

        if 'model1' not in st.session_state or 'model2' not in st.session_state:
            st.warning("⚠️ Please estimate models first (go to VAR/VECM Estimation tab)")
        else:
            # Check if countries have different model types
            model_type1 = st.session_state['model_type1']
            model_type2 = st.session_state['model_type2']
            different_models = model_type1 != model_type2

            # Add comparison option if models differ
            compare_models = False
            if different_models:
                compare_models = st.checkbox(
                    f"📊 Compare forecasts: {country1} ({model_type1}) vs {country2} ({model_type2})",
                    value=False,
                    key='compare_forecasts'
                )

            if compare_models:
                st.info(f"Comparing {country1} ({model_type1}) and {country2} ({model_type2}) forecasts")

                # Generate forecasts for both countries
                with st.spinner("Generating forecasts for both countries..."):
                    try:
                        # Country 1
                        data1_plot = data1.drop('COVID_DUMMY', axis=1) if 'COVID_DUMMY' in data1.columns else data1
                        forecast1_df, lower1_df, upper1_df = st.session_state['model1'].forecast(steps=forecast_periods)

                        # Country 2
                        data2_plot = data2.drop('COVID_DUMMY', axis=1) if 'COVID_DUMMY' in data2.columns else data2
                        forecast2_df, lower2_df, upper2_df = st.session_state['model2'].forecast(steps=forecast_periods)

                        # Find common variables
                        common_vars = list(set(forecast1_df.columns) & set(forecast2_df.columns))

                        # Plot comparison for each variable
                        for var in common_vars:
                            fig = go.Figure()

                            # Country 1 forecast
                            fig.add_trace(go.Scatter(
                                x=list(range(1, forecast_periods + 1)),
                                y=forecast1_df[var],
                                mode='lines',
                                name=f'{country1} ({model_type1})',
                                line=dict(color='blue', width=2)
                            ))

                            # Country 1 confidence interval
                            fig.add_trace(go.Scatter(
                                x=list(range(1, forecast_periods + 1)),
                                y=upper1_df[var],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            fig.add_trace(go.Scatter(
                                x=list(range(1, forecast_periods + 1)),
                                y=lower1_df[var],
                                mode='lines',
                                line=dict(width=0),
                                fillcolor='rgba(0, 0, 255, 0.2)',
                                fill='tonexty',
                                name=f'{country1} 95% CI',
                                hoverinfo='skip'
                            ))

                            # Country 2 forecast
                            fig.add_trace(go.Scatter(
                                x=list(range(1, forecast_periods + 1)),
                                y=forecast2_df[var],
                                mode='lines',
                                name=f'{country2} ({model_type2})',
                                line=dict(color='red', width=2)
                            ))

                            # Country 2 confidence interval
                            fig.add_trace(go.Scatter(
                                x=list(range(1, forecast_periods + 1)),
                                y=upper2_df[var],
                                mode='lines',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            fig.add_trace(go.Scatter(
                                x=list(range(1, forecast_periods + 1)),
                                y=lower2_df[var],
                                mode='lines',
                                line=dict(width=0),
                                fillcolor='rgba(255, 0, 0, 0.2)',
                                fill='tonexty',
                                name=f'{country2} 95% CI',
                                hoverinfo='skip'
                            ))

                            fig.update_layout(
                                title=f'{var} Forecast Comparison: {model_type1} vs {model_type2}',
                                xaxis_title='Periods Ahead',
                                yaxis_title=var,
                                hovermode='x unified',
                                template='plotly_white'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                        # Show comparison table
                        with st.expander("📋 Forecast Comparison Table"):
                            for var in common_vars:
                                st.markdown(f"**{var}**")
                                comparison_df = pd.DataFrame({
                                    'Period': range(1, forecast_periods + 1),
                                    f'{country1} ({model_type1})': forecast1_df[var].values,
                                    f'{country2} ({model_type2})': forecast2_df[var].values,
                                    'Difference': forecast1_df[var].values - forecast2_df[var].values
                                })
                                st.dataframe(comparison_df, use_container_width=True)

                        # Download comparison
                        comparison_data = pd.DataFrame({'Period': range(1, forecast_periods + 1)})
                        for var in common_vars:
                            comparison_data[f'{country1}_{var}_{model_type1}'] = forecast1_df[var].values
                            comparison_data[f'{country2}_{var}_{model_type2}'] = forecast2_df[var].values

                        csv = comparison_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Forecast Comparison (CSV)",
                            data=csv,
                            file_name=f"forecast_comparison_{model_type1}_vs_{model_type2}.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error generating comparison: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                # Original single-country forecast view
                country_choice = st.selectbox("Select Country for Forecast", [country1, country2], key='forecast_country')

                model = st.session_state['model1'] if country_choice == country1 else st.session_state['model2']
                model_type = st.session_state['model_type1'] if country_choice == country1 else st.session_state['model_type2']
                data = data1 if country_choice == country1 else data2

                # Both VAR and VECM support forecasting
                with st.spinner(f"Generating forecasts using {model_type} model..."):
                    try:
                        # Filter out COVID dummy for plotting
                        data_plot = data.drop('COVID_DUMMY', axis=1) if 'COVID_DUMMY' in data.columns else data

                        # Generate forecasts (works for both VAR and VECM)
                        forecast_df, lower_df, upper_df = model.forecast(steps=forecast_periods)

                        # Plot forecasts for each variable
                        for var in data_plot.columns:
                            if var in forecast_df.columns:
                                st.plotly_chart(
                                    plot_forecast(data_plot, forecast_df, lower_df, upper_df, var),
                                    use_container_width=True
                                )

                        # Show forecast table
                        with st.expander("📋 Forecast Values"):
                            forecast_display = forecast_df.copy()
                            forecast_display['Period'] = range(1, len(forecast_display) + 1)
                            forecast_display = forecast_display[['Period'] + list(forecast_df.columns)]
                            st.dataframe(forecast_display, use_container_width=True)

                        # Forecast Error Variance Decomposition
                        st.markdown('<p class="section-header">Forecast Error Variance Decomposition</p>', unsafe_allow_html=True)

                        if model_type == 'VECM':
                            st.info("💡 FEVD for VECM uses the VAR representation in levels")

                        fevd_df = model.fevd(periods=forecast_periods)

                        # Filter variables for selection (exclude COVID dummy)
                        var_options = [col for col in data_plot.columns if col in forecast_df.columns]
                        var_choice = st.selectbox("Select Variable for FEVD", var_options, key='fevd_var')

                        st.plotly_chart(
                            plot_fevd(fevd_df, var_choice, forecast_periods),
                            use_container_width=True
                        )

                        with st.expander("📊 FEVD Table"):
                            fevd_display = fevd_df[fevd_df['Response'] == var_choice].copy()
                            st.dataframe(fevd_display, use_container_width=True)

                        # Download forecast
                        csv = forecast_df.to_csv(index=True)
                        st.download_button(
                            label="📥 Download Forecast Data (CSV)",
                            data=csv,
                            file_name=f"forecast_{country_choice}_{model_type}.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error generating forecasts: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    # TAB 7: Granger Causality
    with tabs[6]:
        st.markdown('<p class="section-header">Granger Causality Tests</p>', unsafe_allow_html=True)

        if 'model1' not in st.session_state or 'model2' not in st.session_state:
            st.warning("⚠️ Please estimate models first (go to VAR/VECM Estimation tab)")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"{country1}")

                model1 = st.session_state['model1']
                model_type1 = st.session_state['model_type1']

                if model_type1 == 'VECM':
                    st.info("💡 VECM Granger causality uses VAR representation in levels")

                with st.spinner(f"Running Granger causality tests for {model_type1}..."):
                    try:
                        maxlag = 8 if model_type1 == 'VAR' else 4  # Fewer lags for VECM
                        gc_results1 = model1.granger_causality(maxlag=maxlag)

                        # Display results for each pair
                        for pair, results_df in gc_results1.items():
                            with st.expander(f"🔗 {pair}"):
                                if 'Error' not in results_df.columns and 'Message' not in results_df.columns:
                                    st.dataframe(results_df, use_container_width=True)

                                    # Highlight if any lag is significant
                                    if 'Significant' in results_df.columns and any(results_df['Significant'] == 'Yes'):
                                        st.success("✅ Statistically significant Granger causality detected")
                                elif 'Message' in results_df.columns:
                                    st.error(results_df['Message'].iloc[0])
                                else:
                                    st.error(results_df.get('Error', ['Unknown error']).iloc[0])

                    except Exception as e:
                        st.error(f"Error in Granger causality test: {str(e)}")

            with col2:
                st.subheader(f"{country2}")

                model2 = st.session_state['model2']
                model_type2 = st.session_state['model_type2']

                if model_type2 == 'VECM':
                    st.info("💡 VECM Granger causality uses VAR representation in levels")

                with st.spinner(f"Running Granger causality tests for {model_type2}..."):
                    try:
                        maxlag = 8 if model_type2 == 'VAR' else 4  # Fewer lags for VECM
                        gc_results2 = model2.granger_causality(maxlag=maxlag)

                        # Display results for each pair
                        for pair, results_df in gc_results2.items():
                            with st.expander(f"🔗 {pair}"):
                                if 'Error' not in results_df.columns and 'Message' not in results_df.columns:
                                    st.dataframe(results_df, use_container_width=True)

                                    # Highlight if any lag is significant
                                    if 'Significant' in results_df.columns and any(results_df['Significant'] == 'Yes'):
                                        st.success("✅ Statistically significant Granger causality detected")
                                elif 'Message' in results_df.columns:
                                    st.error(results_df['Message'].iloc[0])
                                else:
                                    st.error(results_df.get('Error', ['Unknown error']).iloc[0])

                    except Exception as e:
                        st.error(f"Error in Granger causality test: {str(e)}")

    # TAB 8: Historical Decomposition
    with tabs[7]:
        st.markdown('<p class="section-header">Historical Decomposition</p>', unsafe_allow_html=True)

        if 'model1' not in st.session_state or 'model2' not in st.session_state:
            st.warning("⚠️ Please estimate models first (go to VAR/VECM Estimation tab)")
        else:
            country_choice = st.selectbox(
                "Select Country for Historical Decomposition",
                [country1, country2],
                key='hist_decomp_country'
            )

            model = st.session_state['model1'] if country_choice == country1 else st.session_state['model2']
            model_type = st.session_state['model_type1'] if country_choice == country1 else st.session_state['model_type2']
            data = data1 if country_choice == country1 else data2

            # Show info for VECM
            if model_type == 'VECM':
                st.info("ℹ️ Historical decomposition for VECM uses the VAR representation in levels.")

            var_choice = st.selectbox(
                "Select Variable to Decompose",
                data.columns,
                key='hist_decomp_var'
            )

            with st.spinner("Computing historical decomposition..."):
                try:
                    decompositions = model.historical_decomposition()

                    decomp_df = decompositions[var_choice]

                    st.plotly_chart(
                        plot_historical_decomposition(decomp_df, var_choice),
                        use_container_width=True
                    )

                    with st.expander("📋 Decomposition Data"):
                        st.dataframe(decomp_df.tail(20), use_container_width=True)

                    # Download decomposition
                    csv = decomp_df.to_csv(index=True)
                    st.download_button(
                        label="📥 Download Historical Decomposition (CSV)",
                        data=csv,
                        file_name=f"hist_decomp_{country_choice}_{var_choice}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error computing historical decomposition: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # Footer
    st.markdown("---")
    st.markdown("""
    **About this dashboard:**
    - Data source: Federal Reserve Economic Data (FRED)
    - VAR/VECM estimation: statsmodels
    - Visualization: Plotly
    - Framework: Streamlit

    **Note:** This is an educational tool. Results should be validated for research purposes.
    """)


if __name__ == "__main__":
    main()
