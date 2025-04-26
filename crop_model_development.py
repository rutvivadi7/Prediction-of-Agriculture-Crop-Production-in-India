"""
crop_model_development.py
This script handles the development and training of multiple machine learning models 
for crop production prediction in India.
"""

import numpy as np
import pandas as pd
import pickle
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_processed_data():
    """Load the preprocessed training, validation and test datasets."""
    print("Loading processed data...")
    
    X_train = np.load("data/processed/X_train.npy")
    X_val = np.load("data/processed/X_val.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val = np.load("data/processed/y_val.npy")
    y_test = np.load("data/processed/y_test.npy")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, X_train, y_train, model_name):
    """Train a single model and measure training time."""
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    print(f"{model_name} trained in {training_time:.2f} seconds")
    return model, training_time

def evaluate_model(model, X_val, y_val, model_name):
    """Evaluate model on validation data."""
    y_pred = model.predict(X_val)
    
    # Calculate metrics
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"\n{model_name} Validation Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def save_model(model, model_name):
    """Save trained model to disk."""
    filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

def train_all_models(X_train, y_train, X_val, y_val):
    """Train and evaluate multiple models."""
    # Define models to train
    models = {
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': SVR(kernel='rbf', C=10, gamma='scale'),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', max_iter=1000, random_state=42)
    }
    
    # Results storage
    results = []
    trained_models = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        trained_model, train_time = train_model(model, X_train, y_train, name)
        eval_results = evaluate_model(trained_model, X_val, y_val, name)
        eval_results['training_time'] = train_time
        results.append(eval_results)
        trained_models[name] = trained_model
        
        # Save model
        save_model(trained_model, name)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)
    
    # Identify best model
    best_model_idx = results_df['r2'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    print(f"\nBest model based on R² score: {best_model_name}")
    
    return trained_models, results_df, best_model_name

def feature_importance(model, feature_names, model_name):
    """Extract and visualize feature importance if available."""    
    # Extract feature importance if the model supports it
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(indices[:20])), importances[indices[:20]], align='center')
        plt.xticks(range(len(indices[:20])), [feature_names[i] for i in indices[:20]], rotation=90)
        plt.tight_layout()
        plt.savefig(f"visualizations/{model_name.lower().replace(' ', '_')}_feature_importance.png")
        plt.close()
        
        # Return top features
        top_features = [(feature_names[i], importances[i]) for i in indices[:10]]
        return top_features
    else:
        print(f"Feature importance not available for {model_name}")
        return None

def main():
    """Main function to execute the model development pipeline."""
    # Load preprocessed data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Get feature names (placeholder - in real implementation, save these during preprocessing)
    # For demonstration, generating placeholder feature names
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
    
    # Train and evaluate models
    trained_models, results_df, best_model_name = train_all_models(X_train, y_train, X_val, y_val)
    
    # Save comparison results
    results_df.to_csv("results/model_comparison.csv", index=False)
    
    # Get feature importance for tree-based models
    for name, model in trained_models.items():
        if name in ['Random Forest', 'Gradient Boosting']:
            top_features = feature_importance(model, feature_names, name)
            if top_features:
                print(f"\nTop 10 important features for {name}:")
                for feature, importance in top_features:
                    print(f"{feature}: {importance:.4f}")
    
    # Final evaluation on test set
    best_model = trained_models[best_model_name]
    final_results = evaluate_model(best_model, X_test, y_test, f"{best_model_name} (Test Set)")
    
    print("\nModel development completed!")

if __name__ == "__main__":
    main()
