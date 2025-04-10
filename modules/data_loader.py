import pandas as pd
import numpy as np
import io
from utils.validators import validate_required_columns, validate_data_types

def load_data(uploaded_file):
    """
    Load data from uploaded file and perform initial validation
    
    Args:
        uploaded_file: The uploaded file object
        
    Returns:
        Tuple of (DataFrame, list of validation errors)
    """
    validation_errors = []
    
    try:
        # Check file type and read accordingly
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            validation_errors.append(f"Unsupported file type: {file_extension}")
            return None, validation_errors
            
        # Check if the dataframe is empty
        if df.empty:
            validation_errors.append("The uploaded file is empty")
            return df, validation_errors
            
        # Validate required columns
        column_errors = validate_required_columns(df)
        validation_errors.extend(column_errors)
        
        # Validate data types
        datatype_errors = validate_data_types(df)
        validation_errors.extend(datatype_errors)
        
        # Handle Encoded ID as string
        if 'Encoded ID' in df.columns:
            df['Encoded ID'] = df['Encoded ID'].astype(str)
        
        return df, validation_errors
        
    except Exception as e:
        validation_errors.append(f"Error loading file: {str(e)}")
        return None, validation_errors

def validate_data(df):
    """
    Perform comprehensive data validation
    
    Args:
        df: Pandas DataFrame with participant data
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Check for duplicated Encoded IDs
    if 'Encoded ID' in df.columns:
        duplicated_ids = df[df.duplicated('Encoded ID', keep=False)]
        if not duplicated_ids.empty:
            errors.append(f"Found {len(duplicated_ids)} duplicated Encoded IDs")
            
    # Check for missing values in critical fields
    critical_fields = ['Status', 'Encoded ID']
    for field in critical_fields:
        if field in df.columns and df[field].isna().any():
            missing_count = df[field].isna().sum()
            errors.append(f"{missing_count} missing values in {field}")
            
    # Validate Status values
    if 'Status' in df.columns:
        valid_statuses = ['CURRENT-CONTINUING', 'NEW', 'MOVING OUT', 'WAITLIST']
        invalid_statuses = df[~df['Status'].isin(valid_statuses)]['Status'].unique()
        if len(invalid_statuses) > 0:
            errors.append(f"Invalid Status values: {', '.join(map(str, invalid_statuses))}")
    
    # Validate host volunteer information
    if 'host' in df.columns:
        valid_host_values = ['Always', 'Sometimes', 'Never Host', 'n/a', '']
        invalid_host_values = df[~df['host'].isin(valid_host_values)]['host'].unique()
        if len(invalid_host_values) > 0:
            errors.append(f"Invalid host values: {', '.join(map(str, invalid_host_values))}")
    
    return errors
