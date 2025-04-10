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
        
        # Map column names from the data file to the expected column names
        df = map_column_names(df)
        
        # Normalize status values
        df = normalize_status_values(df)
            
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

def map_column_names(df):
    """
    Map column names from the uploaded file to the expected column names
    
    Args:
        df: The original DataFrame with original column names
        
    Returns:
        DataFrame with mapped column names
    """
    # Define column name mapping
    column_mapping = {
        'Alumna Circle Status': 'Status',
        'Requested Region From Form': 'Requested_Region',
        '1st Choice Subregion/ Time Zone': 'first_choice_location',
        '1st Choice Days and Times of Week': 'first_choice_time',
        '2nd Choice Subregion/ Time Zone': 'second_choice_location',
        '2nd Choice Days and Times of Week': 'second_choice_time',
        '3rd Choice Subregion/ Time Zone': 'third_choice_location',
        '3rd Choice Days and Times of Week': 'third_choice_time',
        'Volunteering to Host?': 'host',
        'Current Region': 'Current_Region'
    }
    
    # Create a copy of the DataFrame to avoid modifying the original
    mapped_df = df.copy()
    
    # Rename columns based on the mapping
    for original_col, new_col in column_mapping.items():
        if original_col in mapped_df.columns:
            mapped_df.rename(columns={original_col: new_col}, inplace=True)
    
    return mapped_df

def normalize_status_values(df):
    """
    Normalize the Status values to match the expected format
    
    Args:
        df: DataFrame with Status column
        
    Returns:
        DataFrame with normalized Status values
    """
    if 'Status' not in df.columns:
        return df
    
    # Create a copy of the DataFrame
    normalized_df = df.copy()
    
    # Normalize common variations in Status values
    status_mapping = {
        # Continuing in their circle
        'Current-CONTINUING': 'CURRENT-CONTINUING',
        'Current-CONTINUING ': 'CURRENT-CONTINUING',
        
        # Need new matching (treat these all as NEW)
        'NEW to Circles': 'NEW',
        'Current-MOVING INTO Region': 'NEW',
        'Current - MOVING INTO region': 'NEW',
        'Current-MOVING INTO Region ': 'NEW',
        'Current-MOVING within Region': 'NEW',
        'Current - MOVING OUT OF region': 'NEW',
        'Requesting 2nd Circle': 'NEW',
        'xNEW to Circles (Waitlist)': 'NEW',
        'NEW to Circles (Waitlist) Requesting 2nd Circle': 'NEW',
    }
    
    # Apply the mapping
    normalized_df['Status'] = normalized_df['Status'].apply(
        lambda x: status_mapping.get(x.strip() if isinstance(x, str) else x, x)
    )
    
    # Special handling for "Current-MOVING OUT of Region" - filter these out
    normalized_df = normalized_df[normalized_df['Status'] != 'Current-MOVING OUT of Region']
    
    # Handle region for 'Current-MOVING within Region'
    # For these participants, we should use their Current_Region instead of Requested_Region
    if 'Current_Region' in normalized_df.columns and 'Requested_Region' in normalized_df.columns:
        original_status_col = 'Alumna Circle Status' if 'Alumna Circle Status' in df.columns else 'Status'
        moving_within_mask = df[original_status_col].apply(
            lambda x: isinstance(x, str) and x.strip() == 'Current-MOVING within Region'
        )
        
        # For participants moving within region, use Current_Region as Requested_Region
        normalized_df.loc[moving_within_mask, 'Requested_Region'] = df.loc[moving_within_mask, 'Current_Region']
    
    return normalized_df

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
