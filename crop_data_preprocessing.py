"""
crop_data_preprocessing.py
This script handles data collection, cleaning, and preprocessing for the Indian crop prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the agricultural dataset from CSV file."""
    print(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean the dataset by handling missing values and outliers."""
    print("Cleaning dataset...")
    
    # Drop rows with excessive missing values
    threshold = len(df.columns) * 0.5
    df = df.dropna(thresh=threshold)
    
    # Handle remaining missing values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # For numeric columns, fill missing values with median
    num_imputer = SimpleImputer(strategy='median')
    df[numeric_columns] = num_imputer.fit_transform(df[numeric_columns])
    
    # For categorical columns, fill missing values with mode
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])
    
    # Handle outliers using IQR method for numeric columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def preprocess_features(df):
    """Preprocess features for model training."""
    print("Preprocessing features...")
    
    # Identify feature types
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Remove target column from features
    if 'Production' in numeric_columns:
        numeric_columns = numeric_columns.drop('Production')
    
    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Encode categorical features
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Combine processed features
    processed_df = pd.concat([df[numeric_columns], encoded_df], axis=1)
    
    return processed_df, scaler, encoder

def split_dataset(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Split dataset into training, validation and test sets."""
    print("Splitting dataset...")
    
    # First split: training + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: training vs validation
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    """Main function to execute the data preparation pipeline."""
    # Configuration
    data_path = "data/indian_crop_data.csv"
    
    # Load data
    df = load_data(data_path)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Clean data
    df_cleaned = clean_data(df)
    
    # Extract target variable
    y = df_cleaned['Production']
    
    # Preprocess features
    X, scaler, encoder = preprocess_features(df_cleaned.drop('Production', axis=1))
    
    # Split the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    
    # Save processed data
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_val.npy", X_val)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_val.npy", y_val)
    np.save("data/processed/y_test.npy", y_test)
    
    print("Data preprocessing completed and saved!")

if __name__ == "__main__":
    main()
