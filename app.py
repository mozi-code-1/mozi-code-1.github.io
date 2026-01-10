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

    # TAB 4: Impulse Response Functions
    with tabs[3]:
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

            if model_type == 'VAR':
                with st.spinner("Computing impulse responses..."):
                    try:
                        alpha = (100 - confidence_level) / 100

                        id_scheme = 'cholesky' if 'Cholesky' in identification else 'long_run'

                        irfs, lower, upper = model.irf_with_confidence_bands(
                            periods=irf_periods,
                            alpha=alpha,
                            identification=id_scheme,
                            n_bootstrap=500
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
                            file_name=f"irf_{country_choice}_{shock_var}_{response_var}.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error computing IRFs: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.info("IRF functionality for VECM models coming soon. Currently only available for VAR models.")

    # TAB 5: Forecasting
    with tabs[4]:
        st.markdown('<p class="section-header">Forecasting</p>', unsafe_allow_html=True)

        if 'model1' not in st.session_state or 'model2' not in st.session_state:
            st.warning("⚠️ Please estimate models first (go to VAR/VECM Estimation tab)")
        else:
            country_choice = st.selectbox("Select Country for Forecast", [country1, country2], key='forecast_country')

            model = st.session_state['model1'] if country_choice == country1 else st.session_state['model2']
            model_type = st.session_state['model_type1'] if country_choice == country1 else st.session_state['model_type2']
            data = data1 if country_choice == country1 else data2

            if model_type == 'VAR':
                with st.spinner("Generating forecasts..."):
                    try:
                        forecast_df, lower_df, upper_df = model.forecast(steps=forecast_periods)

                        # Plot forecasts for each variable
                        for var in data.columns:
                            st.plotly_chart(
                                plot_forecast(data, forecast_df, lower_df, upper_df, var),
                                use_container_width=True
                            )

                        # Show forecast table
                        with st.expander("📋 Forecast Values"):
                            forecast_display = forecast_df.copy()
                            forecast_display['Period'] = range(1, len(forecast_display) + 1)
                            forecast_display = forecast_display[['Period'] + list(data.columns)]
                            st.dataframe(forecast_display, use_container_width=True)

                        # Forecast Error Variance Decomposition
                        st.markdown('<p class="section-header">Forecast Error Variance Decomposition</p>', unsafe_allow_html=True)

                        fevd_df = model.fevd(periods=forecast_periods)

                        var_choice = st.selectbox("Select Variable for FEVD", data.columns, key='fevd_var')

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
                            file_name=f"forecast_{country_choice}.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error generating forecasts: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            else:
                st.info("Forecasting functionality for VECM models coming soon. Currently only available for VAR models.")

    # TAB 6: Granger Causality
    with tabs[5]:
        st.markdown('<p class="section-header">Granger Causality Tests</p>', unsafe_allow_html=True)

        if 'model1' not in st.session_state or 'model2' not in st.session_state:
            st.warning("⚠️ Please estimate models first (go to VAR/VECM Estimation tab)")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"{country1}")

                model1 = st.session_state['model1']
                model_type1 = st.session_state['model_type1']

                if model_type1 == 'VAR':
                    with st.spinner("Running Granger causality tests..."):
                        try:
                            gc_results1 = model1.granger_causality(maxlag=min(8, model1.lag_order + 2))

                            # Display results for each pair
                            for pair, results_df in gc_results1.items():
                                with st.expander(f"🔗 {pair}"):
                                    if 'Error' not in results_df.columns:
                                        st.dataframe(results_df, use_container_width=True)

                                        # Highlight if any lag is significant
                                        if any(results_df['Significant'] == 'Yes'):
                                            st.success("✅ Statistically significant Granger causality detected")
                                    else:
                                        st.error(results_df['Error'].iloc[0])

                        except Exception as e:
                            st.error(f"Error in Granger causality test: {str(e)}")
                else:
                    st.info("Granger causality tests not available for VECM models in this version.")

            with col2:
                st.subheader(f"{country2}")

                model2 = st.session_state['model2']
                model_type2 = st.session_state['model_type2']

                if model_type2 == 'VAR':
                    with st.spinner("Running Granger causality tests..."):
                        try:
                            gc_results2 = model2.granger_causality(maxlag=min(8, model2.lag_order + 2))

                            # Display results for each pair
                            for pair, results_df in gc_results2.items():
                                with st.expander(f"🔗 {pair}"):
                                    if 'Error' not in results_df.columns:
                                        st.dataframe(results_df, use_container_width=True)

                                        # Highlight if any lag is significant
                                        if any(results_df['Significant'] == 'Yes'):
                                            st.success("✅ Statistically significant Granger causality detected")
                                    else:
                                        st.error(results_df['Error'].iloc[0])

                        except Exception as e:
                            st.error(f"Error in Granger causality test: {str(e)}")
                else:
                    st.info("Granger causality tests not available for VECM models in this version.")

    # TAB 7: Historical Decomposition
    with tabs[6]:
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

            if model_type == 'VAR':
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
            else:
                st.info("Historical decomposition not available for VECM models in this version.")

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
