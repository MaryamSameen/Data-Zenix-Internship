import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Pakistan Unemployment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b3d9f2;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    # Sample data based on the CSV provided
    data = {
        'Year': list(range(2000, 2024)),
        'Population (millions)': [138.0, 140.5, 143.0, 145.5, 148.0, 150.5, 153.0, 155.5, 158.0, 160.5, 
                                  163.0, 165.5, 168.0, 170.5, 173.0, 175.5, 178.0, 180.5, 183.0, 185.5, 
                                  188.0, 190.5, 193.0, 195.5],
        'GDP Growth Rate (%)': [4.2, 3.1, 3.7, 5.0, 6.4, 7.2, 5.8, 4.8, 1.7, 2.8, 
                                1.6, 2.7, 3.5, 4.0, 4.1, 4.6, 5.5, 5.2, 5.8, 1.9, 
                                -0.5, 3.9, 0.0, 2.5],
        'Inflation Rate (%)': [3.6, 4.4, 3.5, 3.1, 7.4, 9.1, 7.8, 7.0, 20.3, 13.6, 
                               10.1, 11.9, 9.7, 7.4, 8.6, 4.5, 3.8, 4.1, 6.8, 10.7, 
                               9.0, 8.9, 30.8, 11.1],
        'Unemployment Rate (%)': [6.0, 6.2, 6.1, 5.8, 5.5, 5.3, 5.2, 5.5, 5.7, 6.0, 
                                  6.2, 6.3, 6.0, 5.9, 5.8, 5.7, 5.6, 5.4, 5.2, 6.0, 
                                  6.5, 6.3, 5.5, 5.5],
        'Poverty Headcount Ratio (%)': [34.7, 33.5, 32.1, 30.2, 28.6, 27.0, 25.9, 24.7, 23.9, 23.5,
                                         24.0, 23.5, 22.8, 22.3, 21.5, 20.9, 20.1, 19.4, 18.5, 21.9,
                                         22.5, 21.0, 40.0, 40.5],
        'Agriculture Growth Rate (%)': [2.5, 2.0, 4.1, 4.5, 6.0, 5.8, 4.0, 3.5, 1.0, 2.5,
                                        0.5, 3.0, 3.2, 2.9, 2.7, 2.5, 2.0, 2.5, 3.5, 0.5,
                                        -2.0, 3.0, 2.3, 6.25],
        'External Debt (USD billions)': [55.0, 57.0, 58.5, 59.0, 60.0, 61.5, 63.0, 64.5, 66.0, 68.0,
                                         70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0,
                                         90.0, 92.0, 94.0, 96.0],
        'Climate Disasters (count)': [2, 1, 1, 0, 1, 2, 1, 1, 3, 2, 5, 2, 2, 1, 1, 1, 1, 1, 2, 2, 3, 2, 4, 3]
    }
    return pd.DataFrame(data)

def main():
    st.markdown('<h1 class="main-header">🇵🇰 Pakistan Unemployment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Comprehensive analysis of unemployment trends and economic indicators (2000-2023)</p>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar for navigation
    st.sidebar.title("📋 Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Overview", "Unemployment Trends", "Correlation Analysis", "Economic Impact", "Predictive Analysis", "Key Insights"]
    )
    
    if analysis_type == "Overview":
        show_overview(df)
    elif analysis_type == "Unemployment Trends":
        show_unemployment_trends(df)
    elif analysis_type == "Correlation Analysis":
        show_correlation_analysis(df)
    elif analysis_type == "Economic Impact":
        show_economic_impact(df)
    elif analysis_type == "Predictive Analysis":
        show_predictive_analysis(df)
    elif analysis_type == "Key Insights":
        show_key_insights(df)

def show_overview(df):
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📅 Years Covered", f"{df['Year'].min()}-{df['Year'].max()}")
    with col2:
        st.metric("📈 Avg Unemployment", f"{df['Unemployment Rate (%)'].mean():.1f}%")
    with col3:
        st.metric("📊 Max Unemployment", f"{df['Unemployment Rate (%)'].max():.1f}%")
    with col4:
        st.metric("📉 Min Unemployment", f"{df['Unemployment Rate (%)'].min():.1f}%")
    
    # Data preview
    st.markdown('<h3 class="sub-header">📋 Data Preview</h3>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    
    # Basic statistics
    st.markdown('<h3 class="sub-header">📈 Statistical Summary</h3>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2), use_container_width=True)

def show_unemployment_trends(df):
    st.markdown('<h2 class="sub-header">📈 Unemployment Rate Trends</h2>', unsafe_allow_html=True)
    
    # Main unemployment trend
    fig = px.line(df, x='Year', y='Unemployment Rate (%)', 
                  title='Pakistan Unemployment Rate (2000-2023)',
                  markers=True, line_shape='spline')
    
    # Highlight COVID period
    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="red", opacity=0.1, 
                  annotation_text="COVID-19 Period", annotation_position="top left")
    
    fig.update_layout(height=500, showlegend=True)
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)
    
    # Period analysis
    st.markdown('<h3 class="sub-header">📊 Period-wise Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pre-COVID vs COVID comparison
        pre_covid = df[df['Year'] < 2020]['Unemployment Rate (%)'].mean()
        covid_period = df[(df['Year'] >= 2020) & (df['Year'] <= 2021)]['Unemployment Rate (%)'].mean()
        post_covid = df[df['Year'] > 2021]['Unemployment Rate (%)'].mean()
        
        periods_data = pd.DataFrame({
            'Period': ['Pre-COVID (2000-2019)', 'COVID Period (2020-2021)', 'Post-COVID (2022-2023)'],
            'Avg Unemployment (%)': [pre_covid, covid_period, post_covid]
        })
        
        fig_periods = px.bar(periods_data, x='Period', y='Avg Unemployment (%)',
                           title='Average Unemployment by Period',
                           color='Avg Unemployment (%)', color_continuous_scale='Reds')
        st.plotly_chart(fig_periods, use_container_width=True)
    
    with col2:
        # Year-over-year change
        df['Unemployment_Change'] = df['Unemployment Rate (%)'].diff()
        
        fig_change = px.bar(df, x='Year', y='Unemployment_Change',
                          title='Year-over-Year Change in Unemployment Rate',
                          color='Unemployment_Change',
                          color_continuous_scale='RdYlBu_r')
        fig_change.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        st.plotly_chart(fig_change, use_container_width=True)
    
    # Key observations
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**🔍 Key Observations:**")
    st.markdown(f"• Highest unemployment: **{df['Unemployment Rate (%)'].max()}%** in {df.loc[df['Unemployment Rate (%)'].idxmax(), 'Year']}")
    st.markdown(f"• Lowest unemployment: **{df['Unemployment Rate (%)'].min()}%** in {df.loc[df['Unemployment Rate (%)'].idxmin(), 'Year']}")
    st.markdown(f"• COVID-19 impact: Unemployment increased from **{df[df['Year']==2019]['Unemployment Rate (%)'].iloc[0]}%** (2019) to **{df[df['Year']==2020]['Unemployment Rate (%)'].iloc[0]}%** (2020)")
    st.markdown('</div>', unsafe_allow_html=True)

def show_correlation_analysis(df):
    st.markdown('<h2 class="sub-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
    
    # Select numeric columns for correlation
    numeric_cols = ['GDP Growth Rate (%)', 'Inflation Rate (%)', 'Unemployment Rate (%)', 
                   'Poverty Headcount Ratio (%)', 'Agriculture Growth Rate (%)', 
                   'External Debt (USD billions)', 'Climate Disasters (count)']
    
    corr_matrix = df[numeric_cols].corr()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Correlation heatmap
        fig_heatmap = px.imshow(corr_matrix, 
                               text_auto='.2f',
                               title='Correlation Matrix - Economic Indicators',
                               color_continuous_scale='RdBu_r',
                               aspect='auto')
        fig_heatmap.update_layout(height=600)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.markdown('<h4>🎯 Unemployment Correlations</h4>', unsafe_allow_html=True)
        unemployment_corr = corr_matrix['Unemployment Rate (%)'].drop('Unemployment Rate (%)').sort_values(key=abs, ascending=False)
        
        for indicator, correlation in unemployment_corr.items():
            if abs(correlation) > 0.3:
                strength = "Strong" if abs(correlation) > 0.6 else "Moderate"
                direction = "Positive" if correlation > 0 else "Negative"
                st.markdown(f"**{indicator}**: {direction} {strength} ({correlation:.3f})")
    
    # Scatter plots for key relationships
    st.markdown('<h3 class="sub-header">📊 Key Relationships</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GDP Growth vs Unemployment with custom trendline
        fig_scatter1 = px.scatter(df, x='GDP Growth Rate (%)', y='Unemployment Rate (%)',
                                 title='GDP Growth vs Unemployment Rate',
                                 hover_data=['Year'])
        
        # Add custom trendline using numpy
        x_vals = df['GDP Growth Rate (%)'].values
        y_vals = df['Unemployment Rate (%)'].values
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        
        fig_scatter1.add_trace(go.Scatter(x=sorted(x_vals), y=p(sorted(x_vals)),
                                         mode='lines', name=f'Trendline (R²={np.corrcoef(x_vals, y_vals)[0,1]**2:.3f})',
                                         line=dict(dash='dash', color='red')))
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        # Inflation vs Unemployment with custom trendline
        fig_scatter2 = px.scatter(df, x='Inflation Rate (%)', y='Unemployment Rate (%)',
                                 title='Inflation vs Unemployment Rate',
                                 hover_data=['Year'])
        
        # Add custom trendline using numpy
        x_vals2 = df['Inflation Rate (%)'].values
        y_vals2 = df['Unemployment Rate (%)'].values
        z2 = np.polyfit(x_vals2, y_vals2, 1)
        p2 = np.poly1d(z2)
        
        fig_scatter2.add_trace(go.Scatter(x=sorted(x_vals2), y=p2(sorted(x_vals2)),
                                         mode='lines', name=f'Trendline (R²={np.corrcoef(x_vals2, y_vals2)[0,1]**2:.3f})',
                                         line=dict(dash='dash', color='red')))
        st.plotly_chart(fig_scatter2, use_container_width=True)

def show_economic_impact(df):
    st.markdown('<h2 class="sub-header">💰 Economic Impact Analysis</h2>', unsafe_allow_html=True)
    
    # Multi-variable time series
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Unemployment vs GDP Growth', 'Unemployment vs Inflation', 
                       'Unemployment vs Poverty', 'External Debt Trend'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Unemployment vs GDP Growth
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Unemployment Rate (%)'], 
                           name='Unemployment Rate', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Year'], y=df['GDP Growth Rate (%)'], 
                           name='GDP Growth Rate', line=dict(color='blue')), row=1, col=1, secondary_y=True)
    
    # Unemployment vs Inflation
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Unemployment Rate (%)'], 
                           name='Unemployment Rate', showlegend=False, line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Inflation Rate (%)'], 
                           name='Inflation Rate', line=dict(color='orange')), row=1, col=2, secondary_y=True)
    
    # Unemployment vs Poverty
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Unemployment Rate (%)'], 
                           name='Unemployment Rate', showlegend=False, line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Year'], y=df['Poverty Headcount Ratio (%)'], 
                           name='Poverty Rate', line=dict(color='green')), row=2, col=1, secondary_y=True)
    
    # External Debt
    fig.add_trace(go.Scatter(x=df['Year'], y=df['External Debt (USD billions)'], 
                           name='External Debt', line=dict(color='purple')), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Economic Indicators Overview")
    st.plotly_chart(fig, use_container_width=True)
    
    # Economic shocks analysis
    st.markdown('<h3 class="sub-header">⚡ Economic Shocks Impact</h3>', unsafe_allow_html=True)
    
    # Identify recession/crisis periods
    crisis_years = df[df['GDP Growth Rate (%)'] < 2]['Year'].tolist()
    high_inflation_years = df[df['Inflation Rate (%)'] > 15]['Year'].tolist()
    high_unemployment_years = df[df['Unemployment Rate (%)'] > 6]['Year'].tolist()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📉 Low Growth Years**")
        st.markdown(f"Years with GDP growth < 2%: {crisis_years}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**🔥 High Inflation Years**")
        st.markdown(f"Years with inflation > 15%: {high_inflation_years}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📈 High Unemployment Years**")
        st.markdown(f"Years with unemployment > 6%: {high_unemployment_years}")
        st.markdown('</div>', unsafe_allow_html=True)

def show_predictive_analysis(df):
    st.markdown('<h2 class="sub-header">🔮 Predictive Analysis</h2>', unsafe_allow_html=True)
    
    # Prepare features for prediction
    features = ['GDP Growth Rate (%)', 'Inflation Rate (%)', 'Poverty Headcount Ratio (%)', 
               'Agriculture Growth Rate (%)', 'External Debt (USD billions)', 'Climate Disasters (count)']
    
    X = df[features]
    y = df['Unemployment Rate (%)']
    
    # Simple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Actual vs Predicted
        fig_pred = px.scatter(x=y, y=y_pred, title=f'Actual vs Predicted Unemployment Rate (R² = {r2:.3f})')
        fig_pred.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], 
                                     mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
        fig_pred.update_xaxes(title='Actual Unemployment Rate (%)')
        fig_pred.update_yaxes(title='Predicted Unemployment Rate (%)')
        st.plotly_chart(fig_pred, use_container_width=True)
    
    with col2:
        st.markdown('<h4>🎯 Model Performance</h4>', unsafe_allow_html=True)
        st.metric("R² Score", f"{r2:.3f}")
        st.metric("RMSE", f"{np.sqrt(mse):.3f}")
        
        st.markdown('<h4>📊 Feature Importance</h4>', unsafe_allow_html=True)
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        for _, row in feature_importance.iterrows():
            st.markdown(f"**{row['Feature']}**: {row['Coefficient']:.3f}")
    
    # Future prediction scenario
    st.markdown('<h3 class="sub-header">🔭 Scenario Analysis</h3>', unsafe_allow_html=True)
    
    st.markdown("**Adjust economic indicators to see predicted unemployment rate:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gdp_growth = st.slider("GDP Growth Rate (%)", -2.0, 8.0, 3.0, 0.1)
        inflation = st.slider("Inflation Rate (%)", 0.0, 35.0, 8.0, 0.5)
    
    with col2:
        poverty = st.slider("Poverty Rate (%)", 15.0, 45.0, 25.0, 0.5)
        agri_growth = st.slider("Agriculture Growth (%)", -3.0, 7.0, 3.0, 0.1)
    
    with col3:
        debt = st.slider("External Debt (USD billions)", 50.0, 120.0, 85.0, 1.0)
        disasters = st.slider("Climate Disasters", 0, 6, 2, 1)
    
    # Make prediction
    scenario_data = np.array([[gdp_growth, inflation, poverty, agri_growth, debt, disasters]])
    predicted_unemployment = model.predict(scenario_data)[0]
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(f"**🎯 Predicted Unemployment Rate: {predicted_unemployment:.2f}%**")
    
    if predicted_unemployment > 6.0:
        st.markdown("⚠️ **Warning**: High unemployment predicted. Consider policy interventions.")
    elif predicted_unemployment < 5.0:
        st.markdown("✅ **Good**: Low unemployment predicted. Favorable economic conditions.")
    else:
        st.markdown("ℹ️ **Moderate**: Moderate unemployment levels predicted.")
    st.markdown('</div>', unsafe_allow_html=True)

def show_key_insights(df):
    st.markdown('<h2 class="sub-header">🔍 Key Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    # Calculate key statistics
    avg_unemployment = df['Unemployment Rate (%)'].mean()
    unemployment_std = df['Unemployment Rate (%)'].std()
    trend_slope = stats.linregress(df['Year'], df['Unemployment Rate (%)'])[0]
    
    covid_impact = df[df['Year']==2020]['Unemployment Rate (%)'].iloc[0] - df[df['Year']==2019]['Unemployment Rate (%)'].iloc[0]
    
    # Economic correlations
    gdp_corr = df['GDP Growth Rate (%)'].corr(df['Unemployment Rate (%)'])
    inflation_corr = df['Inflation Rate (%)'].corr(df['Unemployment Rate (%)'])
    poverty_corr = df['Poverty Headcount Ratio (%)'].corr(df['Unemployment Rate (%)'])
    
    st.markdown('<h3 class="sub-header">📈 Trend Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**📊 Statistical Overview:**")
        st.markdown(f"• Average unemployment rate: **{avg_unemployment:.1f}%**")
        st.markdown(f"• Standard deviation: **{unemployment_std:.1f}%**")
        st.markdown(f"• Overall trend: **{'+' if trend_slope > 0 else ''}{trend_slope*100:.3f}% per year**")
        st.markdown(f"• COVID-19 impact: **{'+' if covid_impact > 0 else ''}{covid_impact:.1f}% increase**")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown("**🔗 Key Correlations:**")
        st.markdown(f"• GDP Growth: **{gdp_corr:.3f}** ({'Strong' if abs(gdp_corr) > 0.6 else 'Moderate' if abs(gdp_corr) > 0.3 else 'Weak'} {'negative' if gdp_corr < 0 else 'positive'})")
        st.markdown(f"• Inflation: **{inflation_corr:.3f}** ({'Strong' if abs(inflation_corr) > 0.6 else 'Moderate' if abs(inflation_corr) > 0.3 else 'Weak'} {'negative' if inflation_corr < 0 else 'positive'})")
        st.markdown(f"• Poverty: **{poverty_corr:.3f}** ({'Strong' if abs(poverty_corr) > 0.6 else 'Moderate' if abs(poverty_corr) > 0.3 else 'Weak'} {'negative' if poverty_corr < 0 else 'positive'})")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<h3 class="sub-header">🎯 Policy Recommendations</h3>', unsafe_allow_html=True)
    
    recommendations = [
        {
            "title": "🏭 Economic Diversification",
            "description": "Reduce dependence on agriculture by promoting industrial and service sectors to create more employment opportunities."
        },
        {
            "title": "💰 Inflation Control",
            "description": "Implement monetary policies to control inflation, as high inflation periods often coincide with employment challenges."
        },
        {
            "title": "🎓 Skills Development",
            "description": "Invest in education and vocational training programs to match workforce skills with market demands."
        },
        {
            "title": "🌱 Climate Resilience",
            "description": "Build climate-resilient infrastructure and disaster preparedness to minimize economic disruptions from natural disasters."
        },
        {
            "title": "💼 SME Support",
            "description": "Support small and medium enterprises through easier access to credit and reduced bureaucratic barriers."
        }
    ]
    
    for i, rec in enumerate(recommendations):
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.markdown(f"**{rec['title']}**")
        st.markdown(rec['description'])
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Future outlook
    st.markdown('<h3 class="sub-header">🔮 Future Outlook</h3>', unsafe_allow_html=True)
    
    recent_trend = df.tail(5)['Unemployment Rate (%)'].mean()
    historical_avg = df.head(15)['Unemployment Rate (%)'].mean()
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**📊 Outlook Analysis:**")
    if recent_trend > historical_avg:
        st.markdown(f"⚠️ Recent unemployment ({recent_trend:.1f}%) is higher than historical average ({historical_avg:.1f}%)")
        st.markdown("**Priority**: Focus on immediate job creation and economic recovery measures.")
    else:
        st.markdown(f"✅ Recent unemployment ({recent_trend:.1f}%) is lower than historical average ({historical_avg:.1f}%)")
        st.markdown("**Priority**: Maintain current economic policies and focus on sustainable growth.")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()