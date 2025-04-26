"""
crop_model_evaluation.py
This script provides detailed evaluation and visualization of the crop prediction models.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os

def load_test_data():
    """Load the test dataset."""
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy")
    return X_test, y_test

def load_model(model_name):
    """Load a trained model from disk."""
    filename = f"models/{model_name.lower().replace(' ', '_')}.pkl"
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_and_evaluate(model, X_test, y_test, model_name):
    """Make predictions and evaluate the model."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Test Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return y_pred, {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    """Create a scatter plot of actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Add reference line for perfect predictions
    max_val = max(np.max(y_test), np.max(y_pred))
    min_val = min(np.min(y_test), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Actual vs Predicted Crop Production - {model_name}')
    plt.xlabel('Actual Production')
    plt.ylabel('Predicted Production')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"visualizations/{model_name.lower().replace(' ', '_')}_actual_vs_predicted.png")
    plt.close()

def plot_residuals(y_test, y_pred, model_name):
    """Create residual plots to check for patterns."""
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 8))
    
    # Residual plot
    plt.subplot(2, 1, 1)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Residual Plot - {model_name}')
    plt.xlabel('Predicted Production')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # Residual distribution
    plt.subplot(2, 1, 2)
    sns.histplot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Residual Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"visualizations/{model_name.lower().replace(' ', '_')}_residuals.png")
    plt.close()

def plot_error_distribution(y_test, y_pred, model_name):
    """Plot the distribution of prediction errors."""
    errors = np.abs(y_test - y_pred)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title(f'Absolute Error Distribution - {model_name}')
    plt.xlabel('Absolute Error')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"visualizations/{model_name.lower().replace(' ', '_')}_error_distribution.png")
    plt.close()

def evaluate_production_categories(y_test, y_pred, model_name):
    """Evaluate predictions for different production categories."""
    # Define production categories based on quintiles
    quintiles = np.percentile(y_test, [0, 20, 40, 60, 80, 100])
    
    # Function to assign category
    def assign_category(val):
        for i in range(len(quintiles)-1):
            if quintiles[i] <= val < quintiles[i+1]:
                return i
        return len(quintiles)-2  # For the maximum value
    
    # Assign categories
    y_test_cat = np.array([assign_category(y) for y in y_test])
    y_pred_cat = np.array([assign_category(y) for y in y_pred])
    
    # Create confusion matrix
    cm = confusion_matrix(y_test_cat, y_pred_cat)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix of Production Categories - {model_name}')
    plt.xlabel('Predicted Category')
    plt.ylabel('Actual Category')
    category_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    plt.xticks(np.arange(5) + 0.5, category_labels)
    plt.yticks(np.arange(5) + 0.5, category_labels)
    plt.tight_layout()
    plt.savefig(f"visualizations/{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    
    # Calculate classification metrics
    report = classification_report(y_test_cat, y_pred_cat, 
                                 target_names=category_labels, output_dict=True)
    
    # Convert report to DataFrame for saving
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"results/{model_name.lower().replace(' ', '_')}_category_report.csv")
    
    return report_df

def compare_all_models():
    """Compare and visualize performance of all models."""
    # Get list of all model files
    model_files = [f for f in os.listdir("models/") if f.endswith('.pkl')]
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Results storage
    all_results = []
    
    # Evaluate each model
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '').replace('_', ' ').title()
        model = load_model(model_name)
        
        # Make predictions and evaluate
        y_pred, metrics = predict_and_evaluate(model, X_test, y_test, model_name)
        all_results.append(metrics)
        
        # Create visualizations
        plot_actual_vs_predicted(y_test, y_pred, model_name)
        plot_residuals(y_test, y_pred, model_name)
        plot_error_distribution(y_test, y_pred, model_name)
        evaluate_production_categories(y_test, y_pred, model_name)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create comparison bar chart
    plt.figure(figsize=(12, 8))
    
    metrics = ['mse', 'rmse', 'mae']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x='model_name', y=metric, data=results_df)
        plt.title(f'{metric.upper()} Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    # R² comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x='model_name', y='r2', data=results_df)
    plt.title('R² Score Comparison')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("visualizations/model_comparison.png")
    plt.close()
    
    # Save comparison results
    results_df.to_csv("results/final_model_comparison.csv", index=False)
    
    # Return the best model based on R²
    best_model_idx = results_df['r2'].idxmax()
    best_model_name = results_df.loc[best_model_idx, 'model_name']
    
    return best_model_name

def generate_report(best_model_name):
    """Generate a summary report of the evaluation."""
    # Load model comparison results
    comparison_df = pd.read_csv("results/final_model_comparison.csv")
    
    # Create report
    report = f"""
    # Crop Production Prediction Model Evaluation Report
    
    ## Best Performing Model: {best_model_name}
    
    ### Model Performance Metrics
    
    """
    
    # Add best model metrics
    best_model_metrics = comparison_df[comparison_df['model_name'] == best_model_name].iloc[0]
    report += f"- MSE: {best_model_metrics['mse']:.2f}\n"
    report += f"- RMSE: {best_model_metrics['rmse']:.2f}\n"
    report += f"- MAE: {best_model_metrics['mae']:.2f}\n"
    report += f"- R²: {best_model_metrics['r2']:.4f}\n\n"
    
    report += "### All Models Comparison\n\n"
    report += comparison_df.to_markdown(index=False)
    
    # Save report
    with open("results/evaluation_report.md", "w") as f:
        f.write(report)
    
    print(f"Evaluation report generated: results/evaluation_report.md")

def main():
    """Main function to execute the model evaluation pipeline."""
    # Create directories if they don't exist
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Compare all models and get the best one
    best_model_name = compare_all_models()
    print(f"\nBest performing model: {best_model_name}")
    
    # Generate evaluation report
    generate_report(best_model_name)
    
    print("\nModel evaluation completed!")

if __name__ == "__main__":
    main()
