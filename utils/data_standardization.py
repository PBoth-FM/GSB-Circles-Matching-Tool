import pandas as pd
import logging
import ast
from typing import List, Dict, Any, Union, Optional
from utils.feature_flags import get_flag

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_standardization")

def normalize_host_status(status_value: Any) -> str:
    """
    Standardize host status to "ALWAYS", "SOMETIMES", or "NEVER".
    
    Args:
        status_value: The raw host status value which could be in various formats.
        
    Returns:
        str: Standardized host status as "ALWAYS", "SOMETIMES", or "NEVER".
    """
    if not get_flag('use_standardized_host_status'):
        # Simply return the original value converted to string if feature is disabled
        return str(status_value) if status_value is not None else "NEVER"
    
    # Enable verbose logging if debug flag is set
    debug = get_flag('debug_data_standardization')
    if debug:
        logger.info(f"Standardizing host status: '{status_value}' (type: {type(status_value).__name__})")
    
    # Handle None/NaN values
    if status_value is None:
        return "NEVER"
    
    try:
        # For pandas arrays/series and numpy arrays, check if all values are NaN
        if hasattr(pd.isna(status_value), '__iter__'):
            if pd.isna(status_value).all():
                return "NEVER"
        # For scalar values
        elif pd.isna(status_value):
            return "NEVER"
    except Exception as e:
        # If there's any error in checking, log it and try to proceed
        logger.warning(f"Error checking NaN in host status: {str(e)}")
        # Default to NEVER for any errors
        if not status_value:
            return "NEVER"
    
    # Convert to string for consistent processing
    status_str = str(status_value).strip().upper()
    
    # Direct mappings
    if status_str in ["1", "TRUE", "YES", "ALWAYS", "1.0"]:
        return "ALWAYS"
    elif status_str in ["0.5", "SOMETIMES", "ROTATING"]:
        return "SOMETIMES"
    elif status_str in ["0", "FALSE", "NO", "NEVER", "0.0"]:
        return "NEVER"
    
    # Check for boolean values
    if isinstance(status_value, bool):
        return "ALWAYS" if status_value else "NEVER"
    
    # Check for numeric values
    try:
        num_value = float(status_value)
        if num_value == 1.0:
            return "ALWAYS"
        elif num_value == 0.5:
            return "SOMETIMES"
        elif num_value == 0.0:
            return "NEVER"
        else:
            # For any other numeric value between 0 and 1, assume "SOMETIMES"
            if 0 < num_value < 1:
                return "SOMETIMES"
            # For >1, assume "ALWAYS"
            elif num_value > 0:
                return "ALWAYS"
            else:
                return "NEVER"
    except (ValueError, TypeError):
        pass
    
    # Handle "Rotating" or similar terms
    for term in ["ROTAT", "SHAR", "SOME"]:
        if term in status_str:
            return "SOMETIMES"
    
    # Default to "NEVER" for unrecognized values
    if debug:
        logger.warning(f"Unrecognized host status value: '{status_value}', defaulting to 'NEVER'")
    return "NEVER"

def normalize_member_list(members_value: Any) -> List[str]:
    """
    Standardize member lists to a consistent format (List[str]).
    
    Args:
        members_value: The raw members value which could be in various formats:
                       - List of strings or numbers
                       - String representation of a list
                       - Single string or number
                       - None/NaN
        
    Returns:
        List[str]: Standardized list of member IDs as strings.
    """
    if not get_flag('use_standardized_member_lists'):
        # Simply return the original value if feature is disabled
        # but make sure it's a list
        if isinstance(members_value, list):
            return members_value
        elif members_value is None:
            return []
        elif isinstance(members_value, (float, int, str)) and pd.isna(members_value):
            return []
        else:
            return [members_value]
    
    # Enable verbose logging if debug flag is set
    debug = get_flag('debug_data_standardization')
    if debug:
        logger.info(f"Standardizing member list: '{members_value}' (type: {type(members_value).__name__})")
    
    # Handle None/NaN values
    # Use pd.isna().all() for arrays, or direct check for scalar values
    if members_value is None:
        return []
    
    try:
        # For pandas arrays/series and numpy arrays, check if all values are NaN
        if hasattr(pd.isna(members_value), '__iter__'):
            if pd.isna(members_value).all():
                return []
        # For scalar values
        elif pd.isna(members_value):
            return []
    except Exception as e:
        # If there's any error in checking, log it and try to proceed
        logger.warning(f"Error checking NaN in member list: {str(e)}")
        # If we can't determine, but it's clearly empty/None-like, return empty list
        if members_value in (None, [], "", {}):
            return []
    
    # If it's already a list, validate the items
    if isinstance(members_value, list):
        # Convert each element to string and filter out None/NaN
        return [str(member) for member in members_value if not pd.isna(member)]
    
    # If it's a string that looks like a list representation, parse it
    if isinstance(members_value, str):
        if members_value.strip().startswith('[') and members_value.strip().endswith(']'):
            try:
                # Use ast.literal_eval for safe parsing of list literals
                parsed_list = ast.literal_eval(members_value)
                if isinstance(parsed_list, list):
                    # Convert each element to string and filter out None/NaN
                    return [str(member) for member in parsed_list if not pd.isna(member)]
            except (ValueError, SyntaxError):
                if debug:
                    logger.warning(f"Failed to parse string as list: '{members_value}'")
        
        # If it's a comma-separated string, split it
        if ',' in members_value:
            # Split by comma and strip whitespace
            return [item.strip() for item in members_value.split(',') if item.strip()]
        
        # If it's a single string value, return it as a one-item list
        return [members_value.strip()]
    
    # If it's a scalar value (like a number), convert to string and return as list
    return [str(members_value)]

def standardize_circle_metadata(circle_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize all metadata for a single circle.
    
    Args:
        circle_data: Dictionary containing circle metadata.
        
    Returns:
        Dict[str, Any]: Standardized circle metadata.
    """
    # Create a copy of the dictionary to avoid modifying the original
    standardized = circle_data.copy()
    
    # Standardize host statuses
    for field in ['host_status', 'will_host', 'always_hosts', 'sometimes_hosts']:
        if field in standardized:
            standardized[field] = normalize_host_status(standardized[field])
    
    # Standardize member lists
    for field in ['members', 'current_members', 'new_members']:
        if field in standardized:
            standardized[field] = normalize_member_list(standardized[field])
    
    # Ensure consistent ID types (strings)
    if 'circle_id' in standardized:
        standardized['circle_id'] = str(standardized['circle_id'])
    
    return standardized

def standardize_df_host_status(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Standardize host status values in a DataFrame column.
    
    Args:
        df: pandas DataFrame containing host status data
        column_name: Name of the column containing host status values
        
    Returns:
        DataFrame with standardized host status values.
    """
    if column_name not in df.columns:
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Apply standardization function to the column
    result_df[column_name] = result_df[column_name].apply(normalize_host_status)
    
    return result_df

def standardize_df_member_lists(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Standardize member list values in a DataFrame column.
    
    Args:
        df: pandas DataFrame containing member list data
        column_name: Name of the column containing member list values
        
    Returns:
        DataFrame with standardized member list values.
    """
    if column_name not in df.columns:
        return df
    
    # Create a copy of the DataFrame to avoid modifying the original
    result_df = df.copy()
    
    # Apply standardization function to the column
    result_df[column_name] = result_df[column_name].apply(normalize_member_list)
    
    return result_df

def normalize_encoded_id(id_value: Any) -> str:
    """
    Normalize encoded ID values to consistent string format.
    
    Args:
        id_value: The ID value to normalize, could be int, float, or string.
        
    Returns:
        str: Normalized ID as string.
    """
    # Handle None/NaN values
    if id_value is None:
        return ""
    
    try:
        # For pandas arrays/series and numpy arrays, check if all values are NaN
        if hasattr(pd.isna(id_value), '__iter__'):
            if pd.isna(id_value).all():
                return ""
        # For scalar values
        elif pd.isna(id_value):
            return ""
    except Exception as e:
        # If there's any error in checking, log it and try to proceed
        logger.warning(f"Error checking NaN in id value: {str(e)}")
        # If we can't determine, but it seems empty, return empty string
        if not id_value:
            return ""
    
    # Convert to string
    return str(id_value).strip()

def print_normalization_logs(message: str, verbose: bool = False) -> None:
    """
    Print normalization logs if debug flag is enabled.
    
    Args:
        message: The message to log.
        verbose: Whether to print the log even if debug flag is not set.
    """
    debug = get_flag('debug_data_standardization')
    if debug or verbose:
        logger.info(message)

