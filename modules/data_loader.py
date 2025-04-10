import pandas as pd
import numpy as np
import io
from utils.validators import validate_required_columns, validate_data_types

def deduplicate_encoded_ids(df):
    """
    Deduplicate Encoded IDs by adding alphabetical suffixes (A, B, C, etc.)
    
    Args:
        df: DataFrame with potentially duplicate Encoded IDs
        
    Returns:
        Tuple of (DataFrame with unique Encoded IDs, list of messages about changes)
    """
    if 'Encoded ID' not in df.columns:
        return df, []
    
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    deduplication_messages = []
    
    # Ensure Encoded ID is string type
    result_df['Encoded ID'] = result_df['Encoded ID'].astype(str)
    
    # Find duplicate IDs
    duplicate_ids = result_df[result_df.duplicated('Encoded ID', keep=False)]['Encoded ID'].unique()
    
    for duplicate_id in duplicate_ids:
        # Get indexes of all rows with this duplicate ID
        duplicate_indexes = result_df[result_df['Encoded ID'] == duplicate_id].index.tolist()
        
        # Keep the first occurrence as is, modify the rest
        for idx, index in enumerate(duplicate_indexes[1:], 1):
            # Start with 'A' and increment alphabetically
            suffix = chr(64 + idx)  # 'A' for idx=1, 'B' for idx=2, etc.
            
            # Handle case where we might exceed 'Z'
            if idx > 26:
                # For simplicity, if we exceed 26 duplicates, use AA, AB, etc.
                first_char = chr(64 + (idx // 26))
                second_char = chr(64 + (idx % 26 or 26))  # Avoid '@' (ASCII 64)
                suffix = first_char + second_char
            
            # Update the ID with suffix
            new_id = f"{duplicate_id}{suffix}"
            
            # If the new ID already exists (unlikely but possible), increment until we find a unique one
            suffix_idx = 0
            while (result_df['Encoded ID'] == new_id).any():
                suffix_idx += 1
                new_suffix = chr(65 + suffix_idx)  # Start from 'B', 'C', etc.
                new_id = f"{duplicate_id}{new_suffix}"
            
            # Update the ID in the DataFrame
            result_df.at[index, 'Encoded ID'] = new_id
            
            # Create a message about this change
            deduplication_messages.append(f"Encoded ID {duplicate_id} was not unique, split into {duplicate_id} and {new_id}")
    
    return result_df, deduplication_messages

def load_data(uploaded_file):
    """
    Load data from uploaded file and perform initial validation
    
    Args:
        uploaded_file: The uploaded file object
        
    Returns:
        Tuple of (DataFrame, list of validation errors)
    """
    validation_errors = []
    deduplication_messages = []
    
    try:
        # Check file type and read accordingly
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            validation_errors.append(f"Unsupported file type: {file_extension}")
            return None, validation_errors, []
            
        # Check if the dataframe is empty
        if df.empty:
            validation_errors.append("The uploaded file is empty")
            return df, validation_errors, []
        
        # Map column names from the data file to the expected column names
        df = map_column_names(df)
        
        # Normalize status values
        df = normalize_status_values(df)
        
        # Deduplicate Encoded IDs (before other validations)
        if 'Encoded ID' in df.columns:
            df['Encoded ID'] = df['Encoded ID'].astype(str)
            df, deduplication_messages = deduplicate_encoded_ids(df)
            
        # Validate required columns
        column_errors = validate_required_columns(df)
        validation_errors.extend(column_errors)
        
        # Validate data types
        datatype_errors = validate_data_types(df)
        validation_errors.extend(datatype_errors)
        
        return df, validation_errors, deduplication_messages
        
    except Exception as e:
        validation_errors.append(f"Error loading file: {str(e)}")
        return None, validation_errors, []

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
        'Current Region': 'Current_Region',
        'Current Circle ID': 'current_circles_id',
        'Current Subregion': 'Current_Subregion',
        'Current Meeting Time': 'Current_Meeting_Time'
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
