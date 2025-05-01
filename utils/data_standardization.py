import pandas as pd
import logging
from typing import List, Any, Dict, Union, Optional
import ast
import re

logger = logging.getLogger(__name__)

# Track normalization for debugging
host_normalization_log = {}
member_normalization_log = {}

def normalize_host_status(raw_value: Any) -> str:
    """
    Standardize host status values to consistent format: ALWAYS, SOMETIMES, or NEVER
    Also logs the mapping for debugging purposes
    
    Parameters:
    -----------
    raw_value : Any
        The raw host status value from input data
        
    Returns:
    --------
    str
        One of "ALWAYS", "SOMETIMES", or "NEVER"
    """
    # Convert to string representation for consistent logging
    raw_str = str(raw_value) if raw_value is not None else "None"
    
    # Check if we've already processed this value
    if raw_str in host_normalization_log:
        return host_normalization_log[raw_str]
        
    normalized = "NEVER"
    
    # Handle null values
    if pd.isna(raw_value) or raw_value is None:
        normalized = "NEVER"
    else:
        # Normalize to string and lowercase for comparison
        val = str(raw_value).strip().lower()
        
        # Check for "always" variations
        if any(term in val for term in ["always", "always host"]):
            normalized = "ALWAYS"
        # Check for "sometimes" variations  
        elif any(term in val for term in ["sometimes", "sometimes host", "maybe"]):
            normalized = "SOMETIMES"
        # Check for boolean true-like values
        elif val in {"y", "yes", "true", "1", "1.0", "t"}:
            normalized = "ALWAYS"
        # Check for special numeric cases
        elif val.isdigit() and int(val) == 1:
            normalized = "ALWAYS"
    
    # Log the mapping for debugging
    host_normalization_log[raw_str] = normalized
    logger.debug(f"Host normalization: '{raw_value}' → '{normalized}'")
    
    return normalized

def normalize_member_list(value: Any, log_key: Optional[str] = None) -> List[str]:
    """
    Ensures consistent format for member lists as List[str]
    Handles various input formats and logs the transformation
    
    Parameters:
    -----------
    value : Any
        The member list in any format
    log_key : str, optional
        A key to use for logging this normalization (e.g., circle_id)
        
    Returns:
    --------
    List[str]
        Normalized list of member IDs as strings
    """
    original = value  # Save original for logging
    
    # Handle already normalized lists
    if isinstance(value, list):
        # Convert all elements to strings and ensure no empty strings
        result = [str(item).strip() for item in value if str(item).strip()]
        if not result:
            result = []
    
    # Handle null values
    elif pd.isna(value) or value is None:
        result = []
        
    # Handle string values
    elif isinstance(value, str):
        if value.strip() == "":
            result = []
            
        # Check if it's a string representation of a list
        elif value.startswith('[') and value.endswith(']'):
            try:
                # Use AST for safe evaluation
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    result = [str(item).strip() for item in parsed if str(item).strip()]
                else:
                    # Handle case where parsing gives a non-list
                    result = [str(parsed)] if str(parsed).strip() else []
            except (SyntaxError, ValueError):
                # If parsing fails, try other methods
                # Strip brackets and treat as comma-separated
                stripped = value[1:-1].strip()
                if stripped:
                    # Handle quoted items in list strings
                    parts = re.findall(r"'([^']*?)'|\"([^\"]*?)\"|([^,\s]+)", stripped)
                    result = [next(p for p in part if p) for part in parts]
                else:
                    result = []
                
        # Check if it's comma-separated
        elif ',' in value:
            result = [item.strip() for item in value.split(',') if item.strip()]
            
        # It's a single item
        else:
            result = [value.strip()] if value.strip() else []
        
    # Handle numeric or other types
    else:
        result = [str(value)] if str(value).strip() else []
    
    # Log transformation if requested
    if log_key:
        member_normalization_log[log_key] = {
            "original": original,
            "normalized": result
        }
        logger.debug(f"Member normalization for {log_key}: {original} → {result}")
    
    return result

def normalize_encoded_id(value: Any) -> str:
    """
    Ensure Encoded ID is consistently formatted as a string
    
    Parameters:
    -----------
    value : Any
        The Encoded ID in any format
        
    Returns:
    --------
    str
        Normalized Encoded ID as string
    """
    if pd.isna(value) or value is None:
        return ""
    
    return str(value).strip()

def print_normalization_logs():
    """
    Print the host status normalization mapping for debugging
    """
    logger.info("\n🔍 Host Normalization Mapping:")
    for raw, norm in host_normalization_log.items():
        logger.info(f"  '{raw}' → '{norm}'")
    
    logger.info("\n🔍 Member List Normalization Examples:")
    # Print at most 5 examples to avoid overwhelming logs
    for i, (key, mapping) in enumerate(list(member_normalization_log.items())[:5]):
        logger.info(f"  {key}: {mapping['original']} → {mapping['normalized']}")
    
    if len(member_normalization_log) > 5:
        logger.info(f"  ... and {len(member_normalization_log) - 5} more transformations")

def build_circle_metadata(circle_id: str, members: List[str], results_df: pd.DataFrame, target_circle_size: int = 10) -> Dict[str, Any]:
    """
    Build comprehensive circle metadata from member and participant data
    
    Parameters:
    -----------
    circle_id : str
        The ID of the circle
    members : list
        List of member IDs in this circle
    results_df : DataFrame
        The participant data DataFrame
    target_circle_size : int, optional
        The target size for circles (default 10)
        
    Returns:
    --------
    dict
        Complete circle metadata dictionary
    """
    # Ensure members is normalized
    member_ids = normalize_member_list(members, log_key=f"metadata_{circle_id}")
    
    # Clean participant IDs in results
    if "Encoded ID" in results_df.columns:
        results_df_clean = results_df.copy()
        results_df_clean["Encoded ID"] = results_df_clean["Encoded ID"].apply(normalize_encoded_id)
    else:
        results_df_clean = results_df
    
    # Get participant data for these members
    member_data = results_df_clean[results_df_clean["Encoded ID"].isin(member_ids)]
    
    # If we found no member data, log warning and return default metadata
    if len(member_data) == 0:
        logger.warning(f"No member data found for circle {circle_id} with members {member_ids}")
        return {
            "circle_id": circle_id,
            "members": member_ids,
            "member_count": len(member_ids),
            "new_members": 0,
            "continuing_members": 0,
            "always_hosts": 0,
            "sometimes_hosts": 0,
            "max_additions": 0,
            "_warning": "No member data found in results_df"
        }
        
    # Calculate metadata
    # 1. Host counts - using standardized host status
    host_col = "host_status_standardized" if "host_status_standardized" in member_data.columns else next(
        (col for col in member_data.columns if "host" in col.lower()), None)
        
    if host_col:
        # If we're using raw host data, standardize it
        if host_col != "host_status_standardized":
            always_hosts = sum(normalize_host_status(val) == "ALWAYS" 
                              for val in member_data[host_col].values)
            sometimes_hosts = sum(normalize_host_status(val) == "SOMETIMES" 
                                 for val in member_data[host_col].values)
        else:
            # Use already standardized data
            always_hosts = (member_data["host_status_standardized"] == "ALWAYS").sum()
            sometimes_hosts = (member_data["host_status_standardized"] == "SOMETIMES").sum()
    else:
        # No host column found
        always_hosts = 0
        sometimes_hosts = 0
        logger.warning(f"No host column found for circle {circle_id}")
    
    # 2. Determine new vs continuing members
    status_col = next((col for col in member_data.columns if "status" in col.lower()), None)
    if status_col:
        is_continuing = member_data[status_col].str.contains("CONTINUING", case=False, na=False)
        continuing_members = is_continuing.sum()
        new_members = len(member_ids) - continuing_members
    else:
        # No status column found
        continuing_members = 0
        new_members = len(member_ids)
        logger.warning(f"No status column found for circle {circle_id}")
    
    # 3. Calculate max_additions based on continuing members
    max_additions = max(0, target_circle_size - continuing_members)
    
    # 4. Extract region and subregion from circle ID
    region = ""
    subregion = ""
    if circle_id and isinstance(circle_id, str):
        parts = circle_id.split('-')
        if len(parts) >= 2:
            region_code = parts[1]
            # Map region codes to full names
            region_map = {
                'SFO': 'San Francisco',
                'NYC': 'New York',
                'BOS': 'Boston',
                'CHI': 'Chicago',
                # Add other regions as needed
            }
            region = region_map.get(region_code, region_code)
            subregion = region  # Default subregion to region
    
    # 5. Determine meeting time if available
    meeting_time = ""
    meeting_time_col = next((col for col in member_data.columns 
                             if "meeting" in col.lower() and "time" in col.lower()), None)
    if meeting_time_col:
        # Use most common meeting time
        meeting_times = member_data[meeting_time_col].value_counts()
        if not meeting_times.empty:
            meeting_time = meeting_times.index[0]
    
    # 6. Determine if this is a test circle
    is_test_circle = False
    test_circle_patterns = ["TEST", "DEBUG", "SAMPLE"]
    if any(pattern in str(circle_id).upper() for pattern in test_circle_patterns):
        is_test_circle = True
    
    # Build the complete metadata
    metadata = {
        "circle_id": circle_id,
        "members": member_ids,
        "member_count": len(member_ids),
        "new_members": new_members,
        "continuing_members": continuing_members,
        "always_hosts": always_hosts,
        "sometimes_hosts": sometimes_hosts,
        "max_additions": max_additions,
        "region": region,
        "subregion": subregion,
        "meeting_time": meeting_time,
        "is_test_circle": is_test_circle,
        "is_new_circle": continuing_members == 0,
        "is_existing": continuing_members > 0,
        "_created_by": "optimizer_metadata_generator",
        "_timestamp": pd.Timestamp.now().isoformat()
    }
    
    return metadata
