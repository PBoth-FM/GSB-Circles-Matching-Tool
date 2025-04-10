import pandas as pd

def validate_required_columns(df):
    """
    Validate that the DataFrame has the required columns
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        List of error messages for missing columns
    """
    errors = []
    
    # Define required columns
    required_columns = [
        'Encoded ID',
        'Status'
    ]
    
    # Check for missing required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        for col in missing_columns:
            errors.append(f"Required column '{col}' is missing")
    
    # Define recommended columns
    recommended_columns = [
        'Requested_Region',
        'first_choice_location',
        'first_choice_time',
        'second_choice_location',
        'second_choice_time',
        'third_choice_location',
        'third_choice_time',
        'host'
    ]
    
    # Check for missing recommended columns
    missing_recommended = [col for col in recommended_columns if col not in df.columns]
    
    if missing_recommended:
        for col in missing_recommended:
            errors.append(f"Recommended column '{col}' is missing")
    
    return errors

def validate_data_types(df):
    """
    Validate that columns have appropriate data types
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        List of error messages for type issues
    """
    errors = []
    
    # Check Encoded ID can be converted to string
    if 'Encoded ID' in df.columns:
        try:
            df['Encoded ID'].astype(str)
        except Exception as e:
            errors.append(f"Issue with Encoded ID column: {str(e)}")
    
    # Check Status is a valid string value
    if 'Status' in df.columns:
        if df['Status'].dtype != 'object':
            errors.append("Status column should contain text values")
        
        # Check for valid Status values
        valid_statuses = ['CURRENT-CONTINUING', 'NEW', 'MOVING OUT', 'WAITLIST']
        invalid_statuses = df[~df['Status'].isin(valid_statuses)]['Status'].unique()
        
        if len(invalid_statuses) > 0:
            invalid_list = ', '.join([str(s) for s in invalid_statuses])
            errors.append(f"Invalid Status values found: {invalid_list}")
    
    # Check host values if present
    if 'host' in df.columns:
        valid_host_values = ['Always', 'Sometimes', 'Never Host', 'n/a', '']
        
        # Get all non-null and non-empty host values that are not in the valid list
        invalid_hosts = [h for h in df['host'].unique() if pd.notna(h) and h != '' and h not in valid_host_values]
        
        if invalid_hosts:
            invalid_list = ', '.join([str(h) for h in invalid_hosts])
            errors.append(f"Invalid host values found: {invalid_list}")
    
    return errors
