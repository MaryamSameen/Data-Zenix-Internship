import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_explore_data():
    """Load and explore the iris dataset"""
    # Load the iris dataset
    iris = load_iris()
    
    # Create a DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSpecies distribution:")
    print(df['species_name'].value_counts())
    print("\nDataset description:")
    print(df.describe())
    
    return df, iris

def visualize_data(df):
    """Create visualizations of the iris dataset"""
    plt.figure(figsize=(15, 10))
    
    # Pairplot
    plt.subplot(2, 2, 1)
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    
    # Correlation heatmap
    plt.subplot(2, 2, 2)
    correlation_matrix = df[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    
    # Species distribution
    plt.subplot(2, 2, 3)
    df['species_name'].value_counts().plot(kind='bar')
    plt.title('Species Distribution')
    plt.xticks(rotation=45)
    
    # Box plot for petal length by species
    plt.subplot(2, 2, 4)
    df.boxplot(column='petal length (cm)', by='species_name', ax=plt.gca())
    plt.title('Petal Length by Species')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.show()

def train_model(X, y):
    """Train multiple models and select the best one"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return rf_model, scaler, accuracy

def save_model(model, scaler, filename_prefix='iris_model'):
    """Save the trained model and scaler"""
    model_filename = f"{filename_prefix}.pkl"
    scaler_filename = f"{filename_prefix}_scaler.pkl"
    
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    
    print(f"Model saved as {model_filename}")
    print(f"Scaler saved as {scaler_filename}")
    
    return model_filename, scaler_filename

def main():
    """Main function to run the complete pipeline"""
    print("=== Iris Classification Model Training ===\n")
    
    # Load and explore data
    df, iris = load_and_explore_data()
    
    # Visualize data (optional - uncomment if running in Jupyter)
    # visualize_data(df)
    
    # Prepare features and target
    X = iris.data
    y = iris.target
    
    # Train model
    print("\n=== Training Model ===")
    model, scaler, accuracy = train_model(X, y)
    
    # Save model
    print("\n=== Saving Model ===")
    model_file, scaler_file = save_model(model, scaler)
    
    print(f"\n=== Training Complete ===")
    print(f"Final Model Accuracy: {accuracy:.4f}")
    print("Model and scaler saved successfully!")
    
    return model, scaler

if __name__ == "__main__":
    model, scaler = main()