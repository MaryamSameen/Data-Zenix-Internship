import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import time

# Configure page
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
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
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model (with error handling)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("car_price_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'car_price_model.pkl' is in the same directory.")
        return None

model = load_model()

# Sample data for visualizations
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    cars = ["ritz", "sx4", "ciaz", "wagon r", "swift"]
    data = []
    
    for _ in range(200):
        car = np.random.choice(cars)
        year = np.random.randint(2008, 2024)
        present_price = np.random.uniform(10.0, 80.0)  # PKR pricing range
        kms_driven = np.random.randint(5000, 200000)
        fuel_type = np.random.choice(["Petrol", "Diesel", "CNG"])
        selling_price = present_price * (0.6 + 0.3 * np.random.random()) * (1 - (2024 - year) * 0.04)
        
        data.append({
            "Car Name": car,
            "Year": year,
            "Present Price": present_price,
            "KMs Driven": kms_driven,
            "Fuel Type": fuel_type,
            "Selling Price": max(selling_price, 1.0)
        })
    
    return pd.DataFrame(data)

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Interactive Car Price Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.markdown("## 🔧 Car Configuration")
    
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Car details
        st.subheader("🚙 Vehicle Information")
        car_name = st.selectbox(
            "Car Model", 
            ["ritz", "sx4", "ciaz", "wagon r", "swift"],
            help="Select the car model"
        )
        
        year = st.slider(
            "Year of Purchase", 
            2000, 2024, 2015,
            help="Year when the car was purchased"
        )
        
        present_price = st.number_input(
            "Present Price (PKR Lakhs)", 
            min_value=5.0, max_value=200.0, value=25.0, step=0.5,
            help="Current market price of the car in Pakistani Rupees"
        )
        
        kms_driven = st.number_input(
            "Kilometers Driven", 
            min_value=0, max_value=500000, value=30000, step=1000,
            help="Total kilometers driven"
        )
        
        # Car specifications
        st.subheader("⚙️ Specifications")
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        selling_type = st.selectbox("Selling Type", ["Dealer", "Individual", "Trustmark Dealer"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Owner Number", [0, 1, 2, 3], help="Number of previous owners")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Real-time prediction toggle
        real_time = st.checkbox("🔄 Real-time Prediction", value=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Car age and depreciation info
        car_age = 2024 - year
        st.markdown(f"### 📊 Vehicle Analysis")
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Car Age", f"{car_age} years")
        with col_m2:
            st.metric("KMs per Year", f"{int(kms_driven/max(car_age, 1)):,}")
        with col_m3:
            depreciation = (car_age * 8)  # Rough depreciation estimate
            st.metric("Est. Depreciation", f"{min(depreciation, 80)}%")
        with col_m4:
            condition = "Excellent" if kms_driven < 50000 else "Good" if kms_driven < 100000 else "Fair"
            st.metric("Condition", condition)
    
    with col2:
        # Prediction section
        st.markdown("### 🎯 Price Prediction")
        
        # Encode inputs
        if model is not None:
            try:
                fuel_type_encoded = {'Petrol': 2, 'Diesel': 0, 'CNG': 1}[fuel_type]
                selling_type_encoded = {'Dealer': 0, 'Individual': 1, 'Trustmark Dealer': 2}[selling_type]
                transmission_encoded = {'Manual': 1, 'Automatic': 0}[transmission]
                car_name_encoded = {'ritz': 4, 'sx4': 6, 'ciaz': 0, 'wagon r': 8, 'swift': 7}[car_name]
                
                input_data = np.array([[car_name_encoded, year, present_price, kms_driven,
                                        fuel_type_encoded, selling_type_encoded,
                                        transmission_encoded, owner]])
                
                # Real-time or button prediction
                if real_time or st.button("🔮 Predict Price", type="primary"):
                    with st.spinner("Calculating..."):
                        if real_time:
                            time.sleep(0.1)  # Small delay for effect
                        prediction = model.predict(input_data)[0]
                        
                        # Display prediction with styling
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2 style="color: white; margin-bottom: 1rem;">Estimated Selling Price</h2>
                            <h1 style="color: #FFD700; font-size: 2.5rem; margin: 0;">PKR {prediction:.2f} Lakhs</h1>
                            <p style="color: #E8E8E8; margin-top: 1rem;">Based on current market conditions in Pakistan</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Price analysis
                        price_diff = present_price - prediction
                        price_diff_percent = (price_diff / present_price) * 100
                        
                        if price_diff > 0:
                            st.info(f"💡 Expected depreciation: PKR {price_diff:.2f} lakhs ({price_diff_percent:.1f}%)")
                        else:
                            st.success(f"📈 Potential appreciation: PKR {abs(price_diff):.2f} lakhs ({abs(price_diff_percent):.1f}%)")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    # Visualizations section
    st.markdown("---")
    st.markdown("### 📈 Market Analysis Dashboard")
    
    # Generate sample data
    sample_data = generate_sample_data()
    
    # Create visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Price distribution by car model
        fig1 = px.box(
            sample_data, 
            x="Car Name", 
            y="Selling Price",
            title="Price Distribution by Car Model",
            color="Car Name",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with viz_col2:
        # Price vs Year scatter plot
        fig2 = px.scatter(
            sample_data,
            x="Year",
            y="Selling Price",
            color="Car Name",
            size="KMs Driven",
            title="Price Trends Over Years",
            hover_data=["Fuel Type"]
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Additional insights
    viz_col3, viz_col4 = st.columns(2)
    
    with viz_col3:
        # Fuel type distribution
        fuel_counts = sample_data["Fuel Type"].value_counts()
        fig3 = px.pie(
            values=fuel_counts.values,
            names=fuel_counts.index,
            title="Market Share by Fuel Type"
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with viz_col4:
        # Price heatmap by year and car
        pivot_data = sample_data.pivot_table(
            values="Selling Price", 
            index="Car Name", 
            columns="Year", 
            aggfunc="mean"
        )
        fig4 = px.imshow(
            pivot_data,
            title="Average Price Heatmap (Year vs Model)",
            color_continuous_scale="Viridis"
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Market insights
    st.markdown("---")
    st.markdown("### 🔍 Market Insights")
    
    insights_col1, insights_col2, insights_col3 = st.columns(3)
    
    with insights_col1:
        avg_price = sample_data["Selling Price"].mean()
        st.metric("Average Market Price", f"₹{avg_price:.2f} L")
        
    with insights_col2:
        popular_fuel = sample_data["Fuel Type"].mode()[0]
        st.metric("Most Popular Fuel", popular_fuel)
        
    with insights_col3:
        avg_kms = sample_data["KMs Driven"].mean()
        st.metric("Average KMs Driven", f"{avg_kms:,.0f}")
    
    # Tips section
    with st.expander("💡 Tips for Better Car Valuation"):
        st.markdown("""
        - **Service History**: Regular maintenance increases resale value
        - **Accident History**: Clean accident record adds premium
        - **Market Timing**: Festive seasons often see higher prices
        - **Documentation**: Complete papers ensure smooth transactions
        - **Condition**: Interior and exterior condition significantly impact price
        - **Modifications**: Stock condition usually fetches better prices
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "🚗 Car Price Predictor | Built with Streamlit & Machine Learning"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()