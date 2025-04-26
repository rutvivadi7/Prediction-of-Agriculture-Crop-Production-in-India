"""
main.py
Main driver script for the crop production prediction project.
Executes the entire pipeline from data preprocessing to model evaluation.
"""

import os
import argparse
import time
import subprocess
import logging
import json
import pandas as pd
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/execution.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "models",
        "visualizations",
        "results",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory created or verified: {directory}")

def download_dataset(url, output_path):
    """Download dataset from URL if it doesn't exist locally."""
    if os.path.exists(output_path):
        logger.info(f"Dataset already exists at {output_path}")
        return
    
    logger.info(f"Downloading dataset from {url}")
    
    try:
        # Simulating download - in real implementation, use requests or other libraries
        # Example: urllib.request.urlretrieve(url, output_path)
        
        # For demonstration, we'll create a sample dataset
        create_sample_dataset(output_path)
        logger.info(f"Dataset downloaded and saved to {output_path}")
    except Exception as e:
        logger.error(f"Error downloading dataset: {str(e)}")
        raise

def create_sample_dataset(output_path):
    """Create a sample dataset for demonstration purposes."""
    # Create a sample dataframe
    states = ['Andhra Pradesh', 'Bihar', 'Gujarat', 'Karnataka', 'Punjab']
    districts = ['District_' + str(i) for i in range(1, 6)]
    crops = ['Rice', 'Wheat', 'Maize', 'Sugarcane', 'Cotton']
    seasons = ['Kharif', 'Rabi', 'Zaid']
    soil_types = ['Alluvial', 'Black', 'Red', 'Laterite', 'Mountain']
    
    # Create random data
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    
    rows = 1000
    data = {
        'State': np.random.choice(states, size=rows),
        'District': np.random.choice(districts, size=rows),
        'Crop': np.random.choice(crops, size=rows),
        'Season': np.random.choice(seasons, size=rows),
        'Soil_Type': np.random.choice(soil_types, size=rows),
        'Area': np.random.uniform(100, 1000, size=rows),
        'Temperature': np.random.uniform(20, 35, size=rows),
        'Rainfall': np.random.uniform(500, 1500, size=rows),
        'Humidity': np.random.uniform(40, 90, size=rows),
        'pH': np.random.uniform(5.5, 8.5, size=rows),
        'Nitrogen': np.random.uniform(10, 100, size=rows),
        'Phosphorus': np.random.uniform(5, 50, size=rows),
        'Potassium': np.random.uniform(10, 80, size=rows),
        'Production': np.random.uniform(1000, 10000, size=rows)
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation between features and production
    df['Production'] = (
        df['Area'] * 5 +
        (df['Rainfall'] - 500) * 2 +
        (35 - df['Temperature']) * 100 +
        df['Nitrogen'] * 20 +
        np.random.normal(0, 500, size=rows)  # Add some noise
    )
    
    # Make sure production is positive
    df['Production'] = df['Production'].clip(lower=500)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Sample dataset created at {output_path}")

def run_preprocessing():
    """Run the data preprocessing script."""
    logger.info("Starting data preprocessing...")
    start_time = time.time()
    
    try:
        import crop_data_preprocessing
        crop_data_preprocessing.main()
        
        # Alternatively, run as subprocess
        # subprocess.run(["python", "crop_data_preprocessing.py"], check=True)
        
        execution_time = time.time() - start_time
        logger.info(f"Data preprocessing completed in {execution_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def run_model_development():
    """Run the model development script."""
    logger.info("Starting model development...")
    start_time = time.time()
    
    try:
        import crop_model_development
        crop_model_development.main()
        
        # Alternatively, run as subprocess
        # subprocess.run(["python", "crop_model_development.py"], check=True)
        
        execution_time = time.time() - start_time
        logger.info(f"Model development completed in {execution_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in model development: {str(e)}")
        raise

def run_model_evaluation():
    """Run the model evaluation script."""
    logger.info("Starting model evaluation...")
    start_time = time.time()
    
    try:
        import crop_model_evaluation
        crop_model_evaluation.main()
        
        # Alternatively, run as subprocess
        # subprocess.run(["python", "crop_model_evaluation.py"], check=True)
        
        execution_time = time.time() - start_time
        logger.info(f"Model evaluation completed in {execution_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def create_metadata_files():
    """Create necessary metadata files for the web application."""
    logger.info("Creating metadata files...")
    
    # Create feature information
    feature_info = {
        "numerical_features": [
            "Area", "Temperature", "Rainfall", "Humidity", "pH",
            "Nitrogen", "Phosphorus", "Potassium"
        ],
        "categorical_features": [
            "State", "District", "Crop", "Season", "Soil_Type"
        ]
    }
    
    with open("data/feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=4)
    
    # Create states and districts information
    states_districts = {
        "Andhra Pradesh": ["District_1", "District_2"],
        "Bihar": ["District_2", "District_3"],
        "Gujarat": ["District_1", "District_4"],
        "Karnataka": ["District_3", "District_5"],
        "Punjab": ["District_4", "District_5"]
    }
    
    with open("data/states_districts.json", "w") as f:
        json.dump(states_districts, f, indent=4)
    
    # Create crops information
    crops = [
        {"name": "Rice", "season": ["Kharif"]},
        {"name": "Wheat", "season": ["Rabi"]},
        {"name": "Maize", "season": ["Kharif", "Rabi"]},
        {"name": "Sugarcane", "season": ["Zaid"]},
        {"name": "Cotton", "season": ["Kharif"]}
    ]
    
    with open("data/crops.json", "w") as f:
        json.dump(crops, f, indent=4)
    
    logger.info("Metadata files created successfully")

def generate_summary_report():
    """Generate a summary report of the entire pipeline execution."""
    logger.info("Generating summary report...")
    
    try:
        # Load model comparison results
        comparison_df = pd.read_csv("results/final_model_comparison.csv")
        
        # Get best model
        best_model_idx = comparison_df['r2'].idxmax()
        best_model = comparison_df.loc[best_model_idx, 'model_name']
        best_r2 = comparison_df.loc[best_model_idx, 'r2']
        
        # Create report
        report = {
            "execution_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": {
                "rows": 1000,  # In a real implementation, get actual size
                "features": 13
            },
            "best_model": {
                "name": best_model,
                "r2_score": best_r2,
                "rmse": comparison_df.loc[best_model_idx, 'rmse'],
                "mae": comparison_df.loc[best_model_idx, 'mae']
            },
            "all_models": comparison_df.to_dict(orient='records')
        }
        
        # Save report
        with open("results/summary_report.json", "w") as f:
            json.dump(report, f, indent=4)
        
        logger.info("Summary report generated successfully")
        
        # Print summary
        print("\n" + "="*50)
        print(" CROP PREDICTION PROJECT EXECUTION SUMMARY")
        print("="*50)
        print(f"Execution completed at: {report['execution_timestamp']}")
        print(f"Best model: {report['best_model']['name']}")
        print(f"R² score: {report['best_model']['r2_score']:.4f}")
        print(f"RMSE: {report['best_model']['rmse']:.2f}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.error(f"Error generating summary report: {str(e)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run crop production prediction pipeline')
    
    parser.add_argument('--skip-preprocessing', action='store_true', 
                      help='Skip data preprocessing step')
    parser.add_argument('--skip-training', action='store_true', 
                      help='Skip model training step')
    parser.add_argument('--skip-evaluation', action='store_true', 
                      help='Skip model evaluation step')
    parser.add_argument('--data-url', type=str, 
                      default='https://example.com/crop_data.csv',
                      help='URL to download dataset')
    
    return parser.parse_args()

def main():
    """Main function to execute the entire pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    setup_directories()
    
    # Start execution
    logger.info("Starting crop production prediction pipeline")
    total_start_time = time.time()
    
    # Download or verify dataset
    data_path = "data/raw/indian_crop_data.csv"
    download_dataset(args.data_url, data_path)
    
    # Run preprocessing
    if not args.skip_preprocessing:
        run_preprocessing()
    else:
        logger.info("Skipping data preprocessing as requested")
    
    # Run model training
    if not args.skip_training:
        run_model_development()
    else:
        logger.info("Skipping model training as requested")
    
    # Run model evaluation
    if not args.skip_evaluation:
        run_model_evaluation()
    else:
        logger.info("Skipping model evaluation as requested")
    
    # Create metadata files for web application
    create_metadata_files()
    
    # Generate summary report
    generate_summary_report()
    
    # Calculate total execution time
    total_execution_time = time.time() - total_start_time
    logger.info(f"Pipeline execution completed in {total_execution_time:.2f} seconds")

if __name__ == "__main__":
    main()
