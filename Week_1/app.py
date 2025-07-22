# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
import os

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and scaler
@st.cache_resource
def load_model():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('iris_model.pkl')
        scaler = joblib.load('iris_model_scaler.pkl')
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False

def load_iris_data():
    """Load iris dataset for reference"""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]
    return df

def create_feature_plot(df):
    """Create interactive scatter plot of iris features"""
    fig = px.scatter_matrix(
        df,
        dimensions=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
        color='species',
        title="Iris Dataset Feature Relationships",
        color_discrete_map={
            'setosa': '#FF6B6B',
            'versicolor': '#4ECDC4', 
            'virginica': '#45B7D1'
        }
    )
    fig.update_layout(height=600)
    return fig

def create_prediction_gauge(probability, species_name):
    """Create a gauge chart for prediction confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {species_name}"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    # Header
    st.title("🌸 Iris Flower Species Classifier")
    st.markdown("---")
    
    # Load model
    model, scaler, model_loaded = load_model()
    
    if not model_loaded:
        st.error("⚠️ Model files not found! Please run the training script first to generate 'iris_model.pkl' and 'iris_model_scaler.pkl'")
        st.info("Run `python iris_model_training.py` to train and save the model.")
        st.stop()
    
    # Sidebar for input
    with st.sidebar:
        st.header("🔧 Input Features")
        st.markdown("Adjust the measurements below:")
        
        # Input sliders
        sepal_length = st.slider(
            "Sepal Length (cm)", 
            min_value=4.0, max_value=8.0, value=5.8, step=0.1,
            help="Length of the sepal in centimeters"
        )
        
        sepal_width = st.slider(
            "Sepal Width (cm)", 
            min_value=2.0, max_value=5.0, value=3.0, step=0.1,
            help="Width of the sepal in centimeters"
        )
        
        petal_length = st.slider(
            "Petal Length (cm)", 
            min_value=1.0, max_value=7.0, value=4.3, step=0.1,
            help="Length of the petal in centimeters"
        )
        
        petal_width = st.slider(
            "Petal Width (cm)", 
            min_value=0.1, max_value=3.0, value=1.3, step=0.1,
            help="Width of the petal in centimeters"
        )
        
        # Predict button
        predict_button = st.button("🔮 Predict Species", type="primary")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Current Input")
        input_df = pd.DataFrame({
            'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
            'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.dataframe(input_df, use_container_width=True)
        
        # Input visualization
        fig_input = px.bar(
            input_df, x='Feature', y='Value (cm)',
            title="Input Feature Values",
            color='Value (cm)',
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig_input, use_container_width=True)
    
    with col2:
        if predict_button:
            # Make prediction
            input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_scaled = scaler.transform(input_features)
            
            # Get prediction and probabilities
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            species_names = ['Setosa', 'Versicolor', 'Virginica']
            predicted_species = species_names[prediction]
            confidence = probabilities[prediction]
            
            # Display prediction
            st.subheader("🎯 Prediction Result")
            
            # Create prediction display
            if predicted_species == 'Setosa':
                st.success(f"🌺 **{predicted_species}**")
            elif predicted_species == 'Versicolor':
                st.info(f"🌼 **{predicted_species}**")
            else:
                st.warning(f"🌸 **{predicted_species}**")
            
            # Show confidence gauge
            st.plotly_chart(
                create_prediction_gauge(confidence, predicted_species),
                use_container_width=True
            )
            
            # Show all probabilities
            st.subheader("📈 All Probabilities")
            prob_df = pd.DataFrame({
                'Species': species_names,
                'Probability': probabilities,
                'Percentage': [f"{p:.2%}" for p in probabilities]
            }).sort_values('Probability', ascending=False)
            
            st.dataframe(prob_df, use_container_width=True)
            
            # Probability bar chart
            fig_prob = px.bar(
                prob_df, x='Species', y='Probability',
                title="Prediction Probabilities by Species",
                color='Probability',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_prob, use_container_width=True)
    
    # Dataset information
    st.markdown("---")
    st.subheader("📚 About the Dataset")
    
    # Load and display iris data info
    iris_df = load_iris_data()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(iris_df))
    with col2:
        st.metric("Features", 4)
    with col3:
        st.metric("Species", 3)
    
    # Expandable sections
    with st.expander("🔍 View Sample Data"):
        st.dataframe(iris_df.head(10))
    
    with st.expander("📊 Dataset Visualization"):
        fig_scatter = create_feature_plot(iris_df)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with st.expander("📋 Species Information"):
        st.markdown("""
        **Iris Setosa** 🌺
        - Typically has smaller petals
        - Most easily distinguishable species
        - Native to Alaska and northeastern Asia
        
        **Iris Versicolor** 🌼
        - Medium-sized flowers
        - Also known as Blue Flag iris
        - Found in eastern North America
        
        **Iris Virginica** 🌸
        - Largest petals among the three
        - Also called Virginia iris
        - Native to eastern United States
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>Built with Streamlit • Machine Learning Model: Random Forest</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()