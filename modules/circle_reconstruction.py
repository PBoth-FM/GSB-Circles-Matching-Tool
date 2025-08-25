"""
Circle reconstruction utilities to ensure all assigned circles, including post-processed ones,
appear correctly in UI components.
"""

import pandas as pd
import numpy as np
import os

# Cache for normalization tables to avoid repeated loading
_subregion_normalization_cache = None

def load_subregion_normalization_table():
    """
    Load the subregion normalization table from CSV.
    Uses a cache to avoid reloading the file multiple times.
    
    Returns:
        Dictionary with subregion normalization mapping
    """
    global _subregion_normalization_cache
    
    if _subregion_normalization_cache is not None:
        return _subregion_normalization_cache
    
    # Look for the normalization file in possible locations
    possible_paths = [
        "attached_assets/Circles-SubregionNormalization.csv",  # Development path
        "./Circles-SubregionNormalization.csv",                # Root directory
        "../Circles-SubregionNormalization.csv",               # Parent directory
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                print(f"  📂 Loading subregion normalization from {path}")
                norm_df = pd.read_csv(path)
                # Create a mapping dictionary
                mapping = dict(zip(norm_df['All unique variations'], norm_df['Normalized']))
                _subregion_normalization_cache = mapping
                return mapping
            except Exception as e:
                print(f"  ⚠️ Error loading subregion normalization from {path}: {str(e)}")
    
    print("  ⚠️ Could not find subregion normalization table, using identity normalization")
    _subregion_normalization_cache = {}  # Empty mapping as fallback
    return {}

def normalize_subregion(subregion):

def renumber_virtual_circles_by_gmt_offset(circles_df):
    """
    Renumber virtual circles when collapsing different GMT offsets into sequential numbers.
    For example: VO-AE-GMT+3-NEW-01 and VO-AE-GMT+2-NEW-01 become V-AE-NEW-01 and V-AE-NEW-02
    
    Args:
        circles_df: DataFrame containing circle information
        
    Returns:
        DataFrame: Updated DataFrame with renumbered virtual circle IDs
    """
    print("\n🔄 RENUMBERING VIRTUAL CIRCLES: Handling GMT offset removal")
    
    if circles_df is None or circles_df.empty:
        print("  ⚠️ No circles to renumber")
        return circles_df
    
    # Create a mapping of old IDs to new IDs
    circle_id_mapping = {}
    
    # Group virtual circles by region code, separating NEW and continuing
    virtual_circles = {}
    
    for _, row in circles_df.iterrows():
        circle_id = row['circle_id']
        
        if not isinstance(circle_id, str):
            continue
            
        # Only process virtual circles (VO- or V- prefix)
        if not (circle_id.startswith('VO-') or circle_id.startswith('V-')):
            continue
            
        parts = circle_id.split('-')
        
        # Extract region code and determine if it's NEW
        if circle_id.startswith('VO-') and 'GMT' in circle_id:
            # Old format: VO-AM-GMT-5-NEW-01 or VO-AE-GMT+3-01
            region_code = parts[1]  # AM, AE, etc.
            is_new = 'NEW' in parts
            
            # Create grouping key
            group_key = f"V-{region_code}-{'NEW' if is_new else 'CONTINUING'}"
            
            if group_key not in virtual_circles:
                virtual_circles[group_key] = []
            
            # Store original GMT offset for sorting
            gmt_part = None
            for i, part in enumerate(parts):
                if part.startswith('GMT'):
                    if i + 1 < len(parts) and (parts[i + 1].startswith('+') or parts[i + 1].startswith('-')):
                        gmt_part = parts[i + 1]
                    break
            
            virtual_circles[group_key].append({
                'circle_id': circle_id,
                'gmt_offset': gmt_part,
                'row_data': row
            })
    
    # Renumber each group
    for group_key, circles in virtual_circles.items():
        if len(circles) <= 1:
            # No renumbering needed for single circles
            continue
            
        print(f"  📊 Processing group: {group_key} with {len(circles)} circles")
        
        # Sort by GMT offset first for consistent numbering
        def sort_key(circle_info):
            gmt_offset = circle_info['gmt_offset']
            if gmt_offset is None:
                return 0
            try:
                # Convert +3, -5 etc. to numbers for sorting
                return int(gmt_offset)
            except:
                return 0
        
        sorted_circles = sorted(circles, key=sort_key)
        
        # Extract region and NEW status from group key
        group_parts = group_key.split('-')
        region_code = group_parts[1]
        is_new_group = 'NEW' in group_key
        
        # Assign new sequential numbers
        for idx, circle_info in enumerate(sorted_circles, start=1):
            old_id = circle_info['circle_id']
            
            if is_new_group:
                new_id = f"V-{region_code}-NEW-{str(idx).zfill(2)}"
            else:
                new_id = f"V-{region_code}-{str(idx).zfill(2)}"
            
            if old_id != new_id:
                circle_id_mapping[old_id] = new_id
                print(f"    🔄 Renumbering: {old_id} → {new_id}")
    
    # Apply the mapping to the dataframe
    if circle_id_mapping:
        print(f"  🔄 Applying {len(circle_id_mapping)} circle ID changes")
        
        updated_df = circles_df.copy()
        
        # Update circle IDs in the DataFrame
        for old_id, new_id in circle_id_mapping.items():
            updated_df.loc[updated_df['circle_id'] == old_id, 'circle_id'] = new_id
        
        return updated_df
    else:
        print("  ✅ No virtual circles needed renumbering")
        return circles_df


    """
    Normalize a subregion value using the subregion normalization table.
    
    Args:
        subregion: The subregion value to normalize
        
    Returns:
        Normalized subregion value
    """
    if subregion is None:
        return None
    
    # Convert to string if not already
    if not isinstance(subregion, str):
        subregion = str(subregion)
    
    # Load the normalization table
    normalization_table = load_subregion_normalization_table()
    
    # Try to find the value in the normalization table
    normalized_value = normalization_table.get(subregion, subregion)
    
    # Log if we're making a change
    if normalized_value != subregion:
        print(f"  ✅ Normalized subregion from '{subregion}' to '{normalized_value}'")
        
    return normalized_value

def clear_normalization_cache():
    """Clear the normalization cache to force reloading from file."""
    global _subregion_normalization_cache
    _subregion_normalization_cache = None
    print("  🔄 Cleared normalization cache")

def safe_isna(val):
    """
    Enhanced function to safely check if a value is NA, handling all possible pandas objects.
    This function properly handles all edge cases that could cause 'truth value is ambiguous' errors.
    
    Args:
        val: Any value or object to check for NA/NaN status
        
    Returns:
        bool: True if the value is NA/NaN (or all values are NA for collections), False otherwise
    """
    # Add comprehensive type-based debug logging
    val_type = type(val).__name__
    
    # Handle pandas Series
    if isinstance(val, pd.Series):
        # Use .all() to get a single boolean result, even if the Series has multiple values
        result = val.isna().all()
        print(f"  [safe_isna] Series of length {len(val)}, result: {result}")
        return result
        
    # Handle pandas DataFrame
    elif isinstance(val, pd.DataFrame):
        # For DataFrames, check if all values in all columns are NA
        result = val.isna().all().all()
        print(f"  [safe_isna] DataFrame of shape {val.shape}, result: {result}")
        return result
        
    # Handle NumPy arrays
    elif isinstance(val, np.ndarray):
        # For NumPy arrays, use np.isnan and check if all values are NaN
        if val.size == 0:
            # Empty array case
            print(f"  [safe_isna] Empty NumPy array, returning True")
            return True
        
        # Handle different dtypes properly
        if np.issubdtype(val.dtype, np.number):
            result = np.isnan(val).all()
        else:
            # For non-numeric arrays, use pandas isna
            result = all(pd.isna(x) for x in val)
        
        print(f"  [safe_isna] NumPy array of shape {val.shape}, result: {result}")
        return result
        
    # Handle lists
    elif isinstance(val, list):
        if not val:
            # Empty list case
            print(f"  [safe_isna] Empty list, returning True")
            return True
            
        result = all(pd.isna(x) for x in val)
        print(f"  [safe_isna] List of length {len(val)}, result: {result}")
        return result
        
    # Default case for scalar values
    else:
        result = pd.isna(val)
        print(f"  [safe_isna] Scalar value of type {val_type}, result: {result}")
        return result

def renumber_circles_sequentially(circles_df):
    """
    Renumber all circles sequentially within their region code groups.
    This ensures that circle IDs follow a consistent numbered pattern (01, 02, 03...)
    for each region code.
    
    Args:
        circles_df: DataFrame containing circle information
        
    Returns:
        DataFrame: Updated DataFrame with renumbered circle IDs
    """
    print("\n🔄 POST-PROCESSING: Renumbering circles sequentially by region code")
    
    if circles_df is None or circles_df.empty:
        print("  ⚠️ No circles to renumber")
        return circles_df
    
    # Extract all circle IDs
    if 'circle_id' not in circles_df.columns:
        print("  ⚠️ No circle_id column found in DataFrame")
        return circles_df
    
    # Create a mapping of old IDs to new IDs
    circle_id_mapping = {}
    
    # Group circles by region code
    region_groups = {}
    
    for _, row in circles_df.iterrows():
        circle_id = row['circle_id']
        
        if not isinstance(circle_id, str):
            continue
            
        # Skip UNMATCHED circles
        if circle_id == 'UNMATCHED':
            continue
            
        # Extract parts based on circle type (in-person vs virtual)
        parts = circle_id.split('-')
        
        # Skip circles with invalid formats
        if len(parts) < 3:
            continue
            
        # Determine if this is a virtual circle or in-person circle
        is_virtual = parts[0] in ['VO', 'V']
        
        # Extract the region code differently based on circle type
        if is_virtual:
            # Handle new virtual circle format: V-AM-NEW-XX or V-AM-XX
            if len(parts) >= 3:
                region_code = parts[1]  # e.g., AM, AE, etc.
                
                # Check if this is a NEW circle
                is_new = 'NEW' in parts
            else:
                # Unknown format, skip
                continue
        else:
            # Handle in-person circle format: IP-NYC-XX or IP-NYC-NEW-XX
            if len(parts) >= 3:
                region_code = parts[1]  # e.g., NYC, BOS, etc.
                
                # Check if this is a NEW circle
                is_new = 'NEW' in parts
            else:
                # Unknown format, skip
                continue
        
        # Add to the appropriate region group
        key = f"{parts[0]}-{region_code}"  # e.g., IP-NYC or VO-AM-GMT-5
        if key not in region_groups:
            region_groups[key] = []
            
        region_groups[key].append({
            'circle_id': circle_id,
            'is_new': is_new,
            'format_prefix': parts[0],
            'region_code': region_code,
            'row_index': _
        })
    
    # Renumber circles in each region group
    for region_key, circles in region_groups.items():
        # Log the region group
        print(f"  📊 Processing region group: {region_key}")
        
        # Extract format prefix from the key
        format_prefix = region_key.split('-')[0]  # IP or VO
        region_code = region_key[len(format_prefix)+1:]  # Remove the prefix and dash
        
        # Sort existing and new circles separately
        existing_circles = [c for c in circles if not c['is_new']]
        new_circles = [c for c in circles if c['is_new']]
        
        # Process existing circles (no NEW in the ID)
        for idx, circle_info in enumerate(sorted(existing_circles, key=lambda x: x['circle_id']), start=1):
            old_id = circle_info['circle_id']
            
            # Check if this is an existing circle with a non-standard ID format (single digit)
            # For example, IP-ATL-1 instead of IP-ATL-01
            is_continuing_circle = False
            
            # Extract the numeric part of the circle ID
            parts = old_id.split('-')
            if len(parts) >= 3:
                last_part = parts[-1]
                # Check if it's a single digit (e.g., "1" instead of "01")
                if last_part.isdigit() and len(last_part) == 1:
                    is_continuing_circle = True
                    print(f"    🔍 Found continuing circle with non-standard ID format: {old_id}")
            
            if is_continuing_circle:
                # For virtual circles, update to new format
                if format_prefix in ['VO', 'V']:
                    # Convert VO- circles to V- format and remove GMT offset
                    new_id = f"V-{region_code}-{str(idx).zfill(2)}"
                    if old_id != new_id:
                        circle_id_mapping[old_id] = new_id
                        print(f"    🔄 Converting to new format: {old_id} → {new_id}")
                    else:
                        print(f"    ✅ Already in correct format: {old_id}")
                else:
                    # Preserve the original ID format for in-person circles
                    new_id = old_id
                    print(f"    ✅ Preserving original ID format for continuing circle: {old_id}")
            else:
                # Create a new ID with sequential numbering
                # Format: IP-NYC-01 or V-AM-01
                if format_prefix in ['VO', 'V']:
                    new_id = f"V-{region_code}-{str(idx).zfill(2)}"
                else:
                    new_id = f"{format_prefix}-{region_code}-{str(idx).zfill(2)}"
                
                # Only update if the IDs are different
                if old_id != new_id:
                    circle_id_mapping[old_id] = new_id
                    print(f"    🔄 Renumbering: {old_id} → {new_id}")
        
        # Process new circles (with NEW in the ID)
        for idx, circle_info in enumerate(sorted(new_circles, key=lambda x: x['circle_id']), start=1):
            old_id = circle_info['circle_id']
            
            # Create a new ID with sequential numbering
            # Format: IP-NYC-NEW-01 or V-AM-NEW-01
            if format_prefix in ['VO', 'V']:
                new_id = f"V-{region_code}-NEW-{str(idx).zfill(2)}"
            else:
                new_id = f"{format_prefix}-{region_code}-NEW-{str(idx).zfill(2)}"
            
            # Only update if the IDs are different
            if old_id != new_id:
                circle_id_mapping[old_id] = new_id
                print(f"    🔄 Renumbering: {old_id} → {new_id}")
    
    # If there are circles to renumber, update the circle IDs
    if circle_id_mapping:
        print(f"  🔄 Renumbering {len(circle_id_mapping)} circles")
        
        # Create a copy of the DataFrame to avoid modifying while iterating
        updated_df = circles_df.copy()
        
        # Update circle IDs in the DataFrame
        for old_id, new_id in circle_id_mapping.items():
            # Update the circle_id column
            updated_df.loc[updated_df['circle_id'] == old_id, 'circle_id'] = new_id
        
        return updated_df
    else:
        print("  ✅ No circles needed renumbering")
        return circles_df

def fix_virtual_circle_id_format(circle_id, region=None, subregion=None):
    """
    Fix virtual circle IDs that use the incorrect format.
    Transforms V-VIR-NEW-XX to VO-REG-GMT±Y-NEW-XX format.
    
    Args:
        circle_id: The circle ID to fix
        region: Optional region information for the circle
        subregion: Optional subregion information for the circle
        
    Returns:
        str: Fixed circle ID or original if no fix needed
    """
    if not isinstance(circle_id, str):
        return circle_id
        
    # Check if this is an old virtual circle format that needs updating
    if circle_id.startswith('VO-') or circle_id.startswith('V-VIR-'):
        print(f"🔧 UPDATING VIRTUAL CIRCLE ID FORMAT: {circle_id}")
        
        # Extract the parts
        parts = circle_id.split('-')
        
        # Handle different old formats
        if circle_id.startswith('VO-') and 'GMT' in circle_id:
            # Old format: VO-AM-GMT-5-NEW-01 or VO-AE-GMT+3-01
            region_code = parts[1]  # AM, AE, etc.
            
            # Find NEW and number parts
            is_new = 'NEW' in parts
            if is_new:
                new_idx = parts.index('NEW')
                number_part = parts[new_idx + 1] if new_idx + 1 < len(parts) else '01'
                new_id = f"V-{region_code}-NEW-{number_part}"
            else:
                # Find the number (last part that's numeric)
                number_part = next((p for p in reversed(parts) if p.isdigit()), '01')
                new_id = f"V-{region_code}-{number_part}"
            
            print(f"  ✅ Converted {circle_id} to {new_id}")
            return new_id
            
        elif circle_id.startswith('V-VIR-'):
            # Old format: V-VIR-NEW-XX
            if len(parts) >= 3:
                index_str = parts[-1]
                
                # Determine the region code based on the provided region/subregion
                from utils.normalization import get_region_code_with_subregion
                
                if region and subregion:
                    is_virtual = 'Virtual' in str(region) if region is not None else False
                    
                    if is_virtual:
                        # Get proper region code without timezone
                        if 'Americas' in str(region):
                            region_code = 'AM'
                        elif 'APAC' in str(region) or 'EMEA' in str(region):
                            region_code = 'AE'
                        else:
                            region_code = 'AM'  # Default fallback
                        
                        # Create new ID with proper format
                        new_id = f"V-{region_code}-NEW-{index_str}"
                        print(f"  ✅ Converted {circle_id} to {new_id}")
                        return new_id
                else:
                    # If we don't have proper region info, use Americas as default (most common)
                    # This is a fallback and should rarely be used
                    if 'APAC' in str(region) or 'EMEA' in str(region):
                        region_code = 'AE-GMT+1'  # Default for APAC+EMEA
                    else:
                        region_code = 'AM-GMT-5'  # Default for Americas
                        
                    new_id = f"VO-{region_code}-NEW-{index_str}"
                    print(f"  ⚠️ Limited region info - converted {circle_id} to {new_id} using default code")
                    return new_id
        
        # If we couldn't properly convert, return original ID
        return circle_id
    
    # Not a virtual circle or already in correct format
    return circle_id

def reconstruct_circles_from_results(results, original_circles=None, use_standardized_metadata=False):
    """
    Reconstruct circles dataframe from individual participant results.
    This is crucial after post-processing to ensure that all circles, 
    including those with post-processed participants, are properly represented.
    
    This function normalizes all subregion values using the standardized normalization tables.
    
    Args:
        results: List of participant results with assignments
        original_circles: Original circles dataframe (optional)
        use_standardized_metadata: Whether to use standardized metadata from optimizer
        
    Returns:
        DataFrame: Updated circles dataframe with all assigned circles
    """
    # Clear normalization cache to ensure fresh data is loaded
    clear_normalization_cache()
    print("  🔄 Starting circle reconstruction with fresh normalization tables")
    
    # CRITICAL FIX: Detect and fix invalid circle IDs before processing
    fixed_results = []
    circle_id_fixes = {}
    
    for result in results:
        if isinstance(result, dict):
            result_copy = result.copy()
            circle_id = result_copy.get('proposed_NEW_circles_id', '')
            
            # Check if this is an invalid circle ID that needs fixing
            if isinstance(circle_id, str) and ('IP-UNKNOWN' in circle_id or 'IP-Invalid' in circle_id):
                print(f"🔧 CRITICAL FIX: Detected invalid circle ID '{circle_id}'")
                
                # Get participant's region and subregion to determine correct ID
                region = result_copy.get('Derived_Region', result_copy.get('Current_Region', ''))
                subregion = result_copy.get('proposed_NEW_Subregion', result_copy.get('Current_Subregion', ''))
                
                # Generate correct circle ID based on region and subregion
                if 'Virtual' in str(region):
                    # This should be a virtual circle
                    from utils.normalization import get_region_code_with_subregion
                    region_code = get_region_code_with_subregion(region, subregion, is_virtual=True)
                    
                    # Extract the number from the old ID if possible
                    import re
                    number_match = re.search(r'-(\d+)$', circle_id)
                    number = number_match.group(1) if number_match else '01'
                    
                    # Create proper virtual circle ID
                    if 'NEW' in circle_id:
                        new_circle_id = f"VO-{region_code}-NEW-{number}"
                    else:
                        new_circle_id = f"VO-{region_code}-{number}"
                    
                    print(f"  ✅ Fixed virtual circle: {circle_id} → {new_circle_id}")
                    result_copy['proposed_NEW_circles_id'] = new_circle_id
                    circle_id_fixes[circle_id] = new_circle_id
                    
            fixed_results.append(result_copy)
        else:
            fixed_results.append(result)
    
    if circle_id_fixes:
        print(f"🔧 CRITICAL FIX APPLIED: Fixed {len(circle_id_fixes)} invalid circle IDs")
        for old_id, new_id in circle_id_fixes.items():
            print(f"  {old_id} → {new_id}")
    
    # Use the fixed results for further processing
    results = fixed_results
    # REMOVED: Special region mappings that were causing incorrect hardcoded values
    # Instead, we'll rely on actual data from participant records and only use normalization
    # for formatting consistency, not replacing values.
    
    # Define a simple reference of region codes to region names for basic lookups
    # This is only used as a fallback when no region information is available
    REGION_CODE_TO_NAME = {
        'PSA': 'Peninsula',
        'NAP': 'Napa-Sonoma',
        'MXC': 'Mexico City',
        'NBO': 'Nairobi',
        'SAN': 'San Diego',
        'SPO': 'Sao Paulo',
        'NYC': 'New York',
        'BOS': 'Boston',
        'SFO': 'San Francisco',
        'EAB': 'East Bay',
        'ATL': 'Atlanta',
        'MAR': 'Marin County',
        'PAL': 'Palo Alto',
        'SEA': 'Seattle',
        'POR': 'Portland',
        'CHI': 'Chicago',
        'LAX': 'Los Angeles'
    }
    print("\n🔄 RECONSTRUCTING CIRCLES FROM PARTICIPANT RESULTS")
    print("🔍 ENHANCED DIAGNOSTICS: Starting comprehensive circle reconstruction")
    
    # Check if we should use standardized metadata
    from utils.feature_flags import get_flag
    use_standardized_metadata = use_standardized_metadata or get_flag('use_optimizer_metadata')
    if use_standardized_metadata:
        print("  ✅ Using standardized metadata from optimizer")
    
    # Debug any specific test circle IDs to watch
    test_circle_ids = ['IP-WDC-01', 'IP-WDC-02', 'IP-SFO-25', 'IP-SFO-26']
    for circle_id in test_circle_ids:
        print(f"  🔎 TRACKING circle {circle_id} through reconstruction")
    
    # Convert results to DataFrame if it's a list
    if isinstance(results, list):
        # Check first if it's non-empty
        if not results:
            print("  ⚠️ Results list is empty!")
            return pd.DataFrame()
            
        # Handle differences in column naming between result entries
        # Some results may have 'participant_id', others may have 'Encoded ID'
        id_column = 'participant_id'
        if 'Encoded ID' in results[0]:
            id_column = 'Encoded ID'
            
        # Create a dataframe from the results list
        results_df = pd.DataFrame(results)
        print(f"  Created DataFrame with {len(results_df)} participants")
        
        # ENHANCED DIAGNOSTICS: Print column names for debugging
        print(f"  Results DataFrame columns: {results_df.columns.tolist()}")
        
        # Look for meeting time related columns
        meeting_columns = [col for col in results_df.columns if isinstance(col, str) and ('meeting' in col.lower() or 'time' in col.lower() or 'day' in col.lower())]
        print(f"  Meeting-related columns found: {meeting_columns}")
    else:
        # Already a DataFrame
        results_df = results.copy()
        print(f"  Using existing DataFrame with {len(results_df)} participants")
        # Try to determine ID column
        id_column = 'participant_id' if 'participant_id' in results_df.columns else 'Encoded ID'
        
        # ENHANCED DIAGNOSTICS: Print column names for debugging
        print(f"  Results DataFrame columns: {results_df.columns.tolist()}")
    
    # Check for the circle assignment column
    circle_column = None
    for col in ['proposed_NEW_circles_id', 'circle_id', 'assigned_circle']:
        if col in results_df.columns:
            circle_column = col
            break
            
    if not circle_column:
        print("  ⚠️ Could not find circle assignment column!")
        return pd.DataFrame()
        
    print(f"  Using ID column: {id_column}, Circle column: {circle_column}")
    
    # Create a mapping of circle IDs to lists of member IDs
    circle_members = {}
    circle_metadata = {}
    
    # If we're using standardized metadata from the optimizer, we may have it already
    if use_standardized_metadata:
        # Check if there's optimizer metadata in the original circles
        has_optimizer_metadata = False
        if isinstance(original_circles, pd.DataFrame) and not original_circles.empty:
            # Consider ALL circles from the optimizer as having standardized metadata
            # This ensures we use the enhanced metadata from all circle creation points
            if len(original_circles) > 0:
                print(f"  ✅ Using enhanced metadata from {len(original_circles)} optimizer-generated circles")
                has_optimizer_metadata = True
                
                # Use the optimizer's circle metadata as our starting point
                for _, circle in original_circles.iterrows():
                    if 'circle_id' in circle:
                        circle_id = circle['circle_id']
                        circle_metadata[circle_id] = circle.to_dict()
                        
                        # Store the members list separately - with completely separate handling for pandas Series vs scalar values
                        if 'members' in circle:
                            member_val = circle['members']
                            
                            # Use our enhanced safe_isna function to handle all data types safely
                            # First, get the actual value
                            if isinstance(member_val, pd.Series):
                                # For Series objects, get a scalar representation if possible
                                if not member_val.isna().all():  # Safe check using Series method
                                    non_na_values = member_val.dropna()
                                    if len(non_na_values) > 0:
                                        extracted_val = non_na_values.iloc[0]
                                        print(f"  🔍 Extracted scalar value from Series for circle {circle_id}")
                                    else:
                                        print(f"  ⚠️ No non-NA values in Series for circle {circle_id}")
                                        continue
                                else:
                                    print(f"  ⚠️ All values in Series are NA for circle {circle_id}")
                                    continue
                            else:
                                # For non-Series objects, use directly
                                extracted_val = member_val
                            
                            # Now check if the extracted/original value is NA using our safe function
                            if not safe_isna(extracted_val):
                                # Process the value based on its type
                                if isinstance(extracted_val, list):
                                    circle_members[circle_id] = extracted_val
                                    print(f"  ✅ Using list value for circle {circle_id}")
                                elif isinstance(extracted_val, str):
                                    try:
                                        # Try to parse if it's a string representation of a list
                                        import ast
                                        members_list = ast.literal_eval(extracted_val)
                                        if isinstance(members_list, list):
                                            circle_members[circle_id] = members_list
                                            print(f"  ✅ Parsed list from string for circle {circle_id}")
                                    except Exception as e:
                                        print(f"  ⚠️ Could not parse string value for circle {circle_id}: {str(e)}")
                            else:
                                print(f"  ⚠️ Value is NA for circle {circle_id}")
                        
                        # Set region based on region code but always normalize subregions
                        if 'MXC' in circle_id:
                            circle_metadata[circle_id]['region'] = 'Mexico City'
                            # Use normalized value from mapping
                            circle_metadata[circle_id]['subregion'] = normalize_subregion('Mexico City')
                        elif 'NBO' in circle_id:
                            circle_metadata[circle_id]['region'] = 'Nairobi'
                            # Use normalized value from mapping
                            circle_metadata[circle_id]['subregion'] = normalize_subregion('Nairobi')
                        elif 'NAP' in circle_id:
                            circle_metadata[circle_id]['region'] = 'Napa-Sonoma'
                            # Use normalized value from mapping instead of hardcoded 'Napa Valley'
                            circle_metadata[circle_id]['subregion'] = normalize_subregion('Napa/Sonoma')
                        elif 'PSA' in circle_id:
                            circle_metadata[circle_id]['region'] = 'Peninsula'
                            # Normalize existing subregion if present
                            if 'subregion' in circle_metadata[circle_id]:
                                circle_metadata[circle_id]['subregion'] = normalize_subregion(circle_metadata[circle_id]['subregion'])
                        elif 'SPO' in circle_id:
                            circle_metadata[circle_id]['region'] = 'Sao Paulo'
                            # Use normalized value from mapping
                            circle_metadata[circle_id]['subregion'] = normalize_subregion('Sao Paulo')
                        
                        # Always normalize any existing subregion values
                        if 'subregion' in circle_metadata[circle_id]:
                            circle_metadata[circle_id]['subregion'] = normalize_subregion(circle_metadata[circle_id]['subregion'])
        
        if not has_optimizer_metadata:
            print("  ⚠️ No optimizer metadata found in original_circles, proceeding with standard reconstruction")
            use_standardized_metadata = False  # Fall back to standard reconstruction
    
    # Extract all participants assigned to circles (not UNMATCHED)
    matched_df = results_df[results_df[circle_column] != 'UNMATCHED']
    print(f"  Found {len(matched_df)} matched participants")
    
    # Get unique circle IDs
    unique_circles = matched_df[circle_column].unique()
    print(f"  Found {len(unique_circles)} unique circles")
    
    # Extract circle information from the original circles dataframe if provided
    original_circle_info = {}
    if original_circles is not None and isinstance(original_circles, pd.DataFrame):
        for _, row in original_circles.iterrows():
            if 'circle_id' in row:
                c_id = row['circle_id']
                original_circle_info[c_id] = row.to_dict()
        print(f"  Loaded information for {len(original_circle_info)} circles from original data")
        
        # Debug sample of original circle data
        if len(original_circle_info) > 0:
            sample_circle_id = list(original_circle_info.keys())[0]
            sample_data = original_circle_info[sample_circle_id]
            print(f"  Sample original circle {sample_circle_id}:")
            if 'member_count' in sample_data:
                print(f"    member_count: {sample_data['member_count']}")
            if 'members' in sample_data and isinstance(sample_data['members'], list):
                print(f"    members list length: {len(sample_data['members'])}")
    
    # Group participants by circle
    for circle_id in unique_circles:
        # Skip invalid circle IDs
        if safe_isna(circle_id) or circle_id == 'UNMATCHED':
            continue
            
        # Get participants in this circle
        members_df = matched_df[matched_df[circle_column] == circle_id]
        
        # FIX VIRTUAL CIRCLE ID FORMAT - Get region and subregion for the first member
        # to help with conversion
        region = None
        subregion = None
        
        if members_df is not None and not members_df.empty:
            # Try to extract region and subregion from the first member
            first_member = members_df.iloc[0]
            
            # Try different column patterns for region
            for region_col in ['Derived_Region', 'Current_Region', 'region', 'Region']:
                if region_col in first_member and not pd.isna(first_member[region_col]):
                    region = first_member[region_col]
                    break
                    
            # Try different column patterns for subregion
            for subregion_col in ['proposed_NEW_Subregion', 'Current_Subregion', 'subregion', 'Subregion']:
                if subregion_col in first_member and not pd.isna(first_member[subregion_col]):
                    subregion = first_member[subregion_col]
                    break
        
        # Fix virtual circle ID format if needed
        fixed_circle_id = fix_virtual_circle_id_format(circle_id, region, subregion)
        
        # If the circle ID was fixed, update the references
        if fixed_circle_id != circle_id:
            # Update the participants' circle assignments
            matched_df.loc[matched_df[circle_column] == circle_id, circle_column] = fixed_circle_id
            
            # Update the current circle ID we're working with
            circle_id = fixed_circle_id
            print(f"  🔄 Updated circle ID references from {circle_id} to {fixed_circle_id}")
            
            # Re-fetch the members with the updated circle ID
            members_df = matched_df[matched_df[circle_column] == circle_id]
        
        # CRITICAL FIX: Remove duplicate member IDs and filter out invalid IDs (nan, None, empty)
        raw_member_ids = members_df[id_column].tolist()
        
        # Filter out invalid member IDs
        valid_member_ids = []
        invalid_count = 0
        
        for member_id in raw_member_ids:
            # Convert to string for consistent checking
            member_id_str = str(member_id) if member_id is not None else ''
            
            # Check if this is a valid ID
            if (member_id is not None and 
                not pd.isna(member_id) and 
                member_id_str.strip() != '' and
                member_id_str.lower() != 'nan' and
                member_id_str != 'None'):
                valid_member_ids.append(member_id)
            else:
                invalid_count += 1
                
        # Remove duplicates while preserving order
        member_ids_set = set(valid_member_ids)
        member_ids = list(member_ids_set)
        
        if invalid_count > 0:
            print(f"  🧹 Cleaned {invalid_count} invalid member IDs from circle {circle_id}")
            print(f"  📊 Valid unique members: {len(member_ids)} (was {len(raw_member_ids)} total)")
        
        # Log any duplicates removed
        duplicates_removed = len(valid_member_ids) - len(member_ids)
        if duplicates_removed > 0:
            print(f"  🔄 Removed {duplicates_removed} duplicate member IDs from circle {circle_id}")
        
        # Count unique members by status for accurate member counts
        new_members = 0
        continuing_members = 0
        if 'Status' in members_df.columns:
            for _, row in members_df.iterrows():
                member_id = row[id_column]
                # CRITICAL FIX: Handle case where Status might be a float instead of a string
                status_value = row.get('Status', '')
                if isinstance(status_value, (int, float)):
                    # If it's numeric, we can't call upper() - just convert to string
                    status = str(status_value)
                else:
                    # If it's a string, we can safely call upper()
                    status = str(status_value).upper()
                    
                if status == 'NEW' or status == 'NEW TO CIRCLES':
                    new_members += 1
                elif status == 'CURRENT-CONTINUING':
                    continuing_members += 1
        
        print(f"  Circle {circle_id}: Found {len(members_df)} member records, {len(member_ids)} unique members")
        print(f"  Status breakdown: {new_members} new, {continuing_members} continuing members")
        
        if len(members_df) > len(member_ids):
            print(f"  ⚠️ WARNING: Circle {circle_id} had {len(members_df) - len(member_ids)} duplicate member entries - removed")
        
        # Store members list
        circle_members[circle_id] = member_ids
        
        # Initialize circle metadata with the correct member counts
        member_count = len(member_ids)
        
        # CRITICAL FIX: For new circles, make sure we use the actual membership count
        # This ensures new circles show the correct member count
        if circle_id.startswith('IP-NEW'):
            # We should set member_count to the actual number of unique members
            if member_count < len(member_ids):
                member_count = len(member_ids)
                print(f"  🔧 NEW CIRCLE FIX: Set member_count for {circle_id} to {member_count} based on members list")
        has_original_data = circle_id in original_circle_info
        
        # CRITICAL FIX: For continuing circles, prioritize member counts from original data
        # This fixes the issue with continuing circles showing member_count = 1
        if has_original_data and 'member_count' in original_circle_info[circle_id]:
            original_count = original_circle_info[circle_id]['member_count']
            if not pd.isna(original_count) and original_count > member_count:
                print(f"  ✓ Using original member_count={original_count} for circle {circle_id} (instead of {member_count})")
                member_count = original_count
            
        # Initialize circle metadata 
        circle_metadata[circle_id] = {
            'circle_id': circle_id,
            'member_count': member_count,
            'members': member_ids
        }
        
        # Try to extract region, subregion, meeting time from results
        # IMPROVED: Instead of just using the first member, we'll check all members for metadata
        # This ensures we use the most complete data available for continuing circles
        
        # CRITICAL FIX: Check if this is a virtual or new circle to apply appropriate logic
        is_virtual = isinstance(circle_id, str) and ('VO-' in circle_id or 'V-VIR' in circle_id)
        is_new_circle = isinstance(circle_id, str) and ('NEW' in circle_id or circle_id.startswith('IP-NEW') or circle_id.startswith('V-VIR-NEW'))
        is_continuing_circle = not is_new_circle and any('CURRENT-CONTINUING' == str(row.get('Status', '')).upper() for _, row in members_df.iterrows())
        
        # Display circle type for debugging
        print(f"  Circle {circle_id}: Type: {'NEW' if is_new_circle else 'CONTINUING' if is_continuing_circle else 'UNKNOWN'}, Virtual: {is_virtual}")
        
        # Define property extraction for each member
        property_values = {
            'region': [],
            'subregion': [],
            'meeting_day': [],
            'meeting_time': [],
            'meeting_day_time': []  # For the combined proposed_NEW_DayTime field
        }
        
        # Define column priorities based on circle type
        # For region columns, always prioritize Derived_Region first
        # For new circles, prioritize proposed_NEW fields over Current fields
        property_columns = {
            'region': ['Derived_Region', 'Current_Region', 'proposed_NEW_Region', 'region'],
            'subregion': ['proposed_NEW_Subregion', 'Current_Subregion', 'subregion'],
            'meeting_day': ['Current_Meeting_Day', 'Current Meeting Day', 'Current/ Continuing Meeting Day',
                          'Meeting Day', 'Preferred Meeting Day'],
            'meeting_time': ['Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time', 
                           'Meeting Time', 'Preferred Meeting Time'],
            'meeting_day_time': ['proposed_NEW_DayTime', 'Current_DayTime']
        }
        
        # Check every member for each property
        for _, member in members_df.iterrows():
            # Get member status
            status = str(member.get('Status', '')).upper() if not pd.isna(member.get('Status', '')) else ''
            
            # For continuing circles, prioritize data from CURRENT-CONTINUING members
            for prop, columns in property_columns.items():
                for col in columns:
                    if col in member and not pd.isna(member[col]) and member[col]:
                        # Store the value and the member's status (for prioritization)
                        property_values[prop].append({
                            'value': str(member[col]),
                            'is_continuing': status == 'CURRENT-CONTINUING',
                            'column': col
                        })
                        break
        
        # Process the collected values for each property
        extracted_props = {}
        
        # NEW CODE: First check if this is a new circle and try to extract the proposed values directly
        if is_new_circle:
            print(f"  🔍 NEW CIRCLE DETECTED: {circle_id} - Looking for proposed_NEW values first")
            
            # For region, prioritize Derived_Region
            derived_region_found = False
            for _, member in members_df.iterrows():
                if 'Derived_Region' in member and not pd.isna(member['Derived_Region']) and member['Derived_Region']:
                    extracted_props['region'] = str(member['Derived_Region'])
                    print(f"  ✅ NEW CIRCLE: Set region='{extracted_props['region']}' from Derived_Region")
                    derived_region_found = True
                    break
            
            # If Derived_Region not found, use Requested_Region
            if not derived_region_found:
                for _, member in members_df.iterrows():
                    if 'Requested_Region' in member and not pd.isna(member['Requested_Region']) and member['Requested_Region']:
                        extracted_props['region'] = str(member['Requested_Region'])
                        print(f"  ✅ NEW CIRCLE: Set region='{extracted_props['region']}' from Requested_Region")
                        break
            
            # For subregion, use proposed_NEW_Subregion
            proposed_subregion_found = False
            for _, member in members_df.iterrows():
                if 'proposed_NEW_Subregion' in member and not pd.isna(member['proposed_NEW_Subregion']) and member['proposed_NEW_Subregion']:
                    # Fix for Peninsula circles - Check if the proposed value incorrectly contains "Phoenix/Scottsdale/Arizona"
                    proposed_value = str(member['proposed_NEW_Subregion'])
                    
                    # Special handling for Peninsula (PSA) circles - convert Phoenix/Scottsdale/Arizona back to correct subregion
                    if 'Phoenix/Scottsdale/Arizona' in proposed_value and (
                        circle_id.startswith('IP-PSA-') or 
                        'Peninsula' in member.get('Current_Region', '') or 
                        'Peninsula' in member.get('Derived_Region', '') or
                        'Peninsula' in member.get('Requested_Region', '')):
                        
                        print(f"  🛠️ FIXING: Detected incorrect 'Phoenix/Scottsdale/Arizona' for Peninsula circle {circle_id}")
                        
                        # Try to get the correct subregion from Circle-RegionSubregionCodeMapping
                        correct_subregion = None
                        
                        # Look for correct subregion in other columns first
                        for subregion_col in ['Current_Subregion', 'Requested_Subregion']:
                            if subregion_col in member and not pd.isna(member[subregion_col]) and member[subregion_col]:
                                correct_subregion = str(member[subregion_col])
                                print(f"  ✅ FIXED: Found correct subregion '{correct_subregion}' in {subregion_col}")
                                break
                        
                        # If no correct subregion found, use a standardized approach with normalized values from mapping
                        if not correct_subregion:
                            # Extract circle number to determine subregion consistently
                            if circle_id and '-' in circle_id:
                                parts = circle_id.split('-')
                                if len(parts) >= 3 and parts[-1].isdigit():
                                    circle_num = int(parts[-1])
                                    
                                    # Get region code from circle ID
                                    region_code = 'PSA' # Default for Peninsula
                                    if len(parts) >= 2:
                                        region_code = parts[1]
                                    
                                    # IMPROVED: Use actual subregion data or a generic fallback
                                    # Try to find actual subregion data in members of this circle first
                                    from_data = members_df.copy()
                                    if not from_data.empty and 'Current_Subregion' in from_data.columns:
                                        # Look for non-null subregions from existing members
                                        valid_subregions = from_data[from_data['Current_Subregion'].notna()]['Current_Subregion'].unique()
                                        if len(valid_subregions) > 0:
                                            # Use the first available subregion
                                            correct_subregion = str(valid_subregions[0])
                                            print(f"  ✅ Using ACTUAL subregion '{correct_subregion}' from circle member data")
                                        else:
                                            # No data available, use generic fallback
                                            correct_subregion = "Unknown"
                                            print(f"  ⚠️ No actual subregion data found, using 'Unknown'")
                                    else:
                                        # No member data available, use region name + 'Area' as fallback
                                        region_name = REGION_CODE_TO_NAME.get(region_code, "Unknown Region")
                                        correct_subregion = "Unknown"
                                        print(f"  ⚠️ No member data available, using 'Unknown' for {region_name}")
                                else:
                                    # If no circle number, use a generic value
                                    correct_subregion = "Unknown"
                                    print(f"  ⚠️ No valid circle number, using 'Unknown' as subregion")
                        
                        # Use the corrected subregion
                        extracted_props['subregion'] = correct_subregion
                    else:
                        # Normal case (non-Peninsula or already correct)
                        extracted_props['subregion'] = proposed_value
                    
                    print(f"  ✅ NEW CIRCLE: Set subregion='{extracted_props['subregion']}' from proposed_NEW_Subregion")
                    proposed_subregion_found = True
                    break
            
            # For meeting_time, use proposed_NEW_DayTime
            proposed_time_found = False
            for _, member in members_df.iterrows():
                if 'proposed_NEW_DayTime' in member and not pd.isna(member['proposed_NEW_DayTime']) and member['proposed_NEW_DayTime']:
                    proposed_value = str(member['proposed_NEW_DayTime'])
                    
                    # Special handling for Peninsula (PSA) circles with Unknown meeting time
                    if (proposed_value == 'Unknown' or not proposed_value) and (
                        circle_id.startswith('IP-PSA-') or 
                        'Peninsula' in member.get('Current_Region', '') or 
                        'Peninsula' in member.get('Derived_Region', '') or
                        'Peninsula' in member.get('Requested_Region', '')):
                        
                        print(f"  🛠️ FIXING: Detected 'Unknown' meeting time for Peninsula circle {circle_id}")
                        
                        # Try to get the correct meeting time from other columns first
                        correct_time = None
                        
                        # Look for meeting time in these priority columns
                        for time_col in ['Current_DayTime', 'Requested_DayTime', 'Preference_1_DayTime']:
                            if time_col in member and not pd.isna(member[time_col]) and member[time_col]:
                                correct_time = str(member[time_col])
                                print(f"  ✅ FIXED: Found correct meeting time '{correct_time}' in {time_col}")
                                break
                        
                        # If no correct time found, use consistent meeting times from region mappings
                        if not correct_time:
                            # Extract region code and circle number to determine meeting time consistently
                            if circle_id and '-' in circle_id:
                                parts = circle_id.split('-')
                                if len(parts) >= 3 and parts[-1].isdigit():
                                    circle_num = int(parts[-1])
                                    
                                    # Get region code from circle ID
                                    region_code = 'PSA' # Default for Peninsula
                                    if len(parts) >= 2:
                                        region_code = parts[1]
                                    
                                    # IMPROVED: Use actual meeting time data or a generic fallback
                                    # Try to find actual meeting time data in members of this circle first
                                    from_data = members_df.copy()
                                    if not from_data.empty and 'Current_DayTime' in from_data.columns:
                                        # Look for non-null meeting times from existing members
                                        valid_times = from_data[from_data['Current_DayTime'].notna()]['Current_DayTime'].unique()
                                        if len(valid_times) > 0:
                                            # Use the first available meeting time
                                            correct_time = str(valid_times[0])
                                            print(f"  ✅ Using ACTUAL meeting time '{correct_time}' from circle member data")
                                        else:
                                            # No data available, use generic fallback
                                            correct_time = "Varies (Evenings)"
                                            print(f"  ⚠️ No actual meeting time found, using 'Varies (Evenings)'")
                                    else:
                                        # No member data available, use generic fallback
                                        correct_time = "Varies (Evenings)"
                                        print(f"  ⚠️ No member data available, using 'Varies (Evenings)' for meeting time")
                                else:
                                    # If no circle number, use generic meeting time
                                    correct_time = "Varies (Evenings)"
                                    print(f"  ⚠️ No valid circle number, using default meeting time")
                            else:
                                # No valid circle ID format - use default time from PSA
                                correct_time = "Wednesday (Evenings)"  # Most common meeting time
                                print(f"  📊 Fallback: Using standard meeting time 'Wednesday (Evenings)' for circle")
                        
                        # Use the corrected meeting time
                        extracted_props['meeting_time'] = correct_time
                    else:
                        # Normal case (non-Peninsula or already correct)
                        extracted_props['meeting_time'] = proposed_value
                    
                    print(f"  ✅ NEW CIRCLE: Set meeting_time='{extracted_props['meeting_time']}' from proposed_NEW_DayTime")
                    proposed_time_found = True
                    break
            
            # Virtual circles need special handling to ensure region is set correctly
            if is_virtual and 'region' in extracted_props:
                if 'Virtual' not in extracted_props['region']:
                    # Update region to include Virtual designation if not already there
                    if 'Americas' in extracted_props['region']:
                        extracted_props['region'] = 'Virtual-Only Americas'
                    elif 'APAC' in extracted_props['region'] or 'EMEA' in extracted_props['region']:
                        extracted_props['region'] = 'Virtual-Only APAC+EMEA'
                    else:
                        # Generic virtual designation as fallback
                        extracted_props['region'] = f"Virtual-Only {extracted_props['region']}"
                    print(f"  ✅ VIRTUAL CIRCLE: Updated region to '{extracted_props['region']}'")
        
        # If we're missing information, fall back to the standard approach for all circles
        # For region and subregion, prioritize values from CURRENT-CONTINUING members if not already set
        for prop in ['region', 'subregion']:
            # Skip if already set for new circles
            if prop in extracted_props and extracted_props[prop] and extracted_props[prop] != 'Unknown':
                continue
                
            if property_values[prop]:
                # First try CURRENT-CONTINUING members' values
                continuing_values = [item['value'] for item in property_values[prop] if item['is_continuing']]
                if continuing_values:
                    extracted_props[prop] = continuing_values[0]  # Use the first non-empty value
                    print(f"  ✅ Set {prop}='{extracted_props[prop]}' from CURRENT-CONTINUING member")
                else:
                    # If no CURRENT-CONTINUING values, use any value
                    extracted_props[prop] = property_values[prop][0]['value']
                    print(f"  ✅ Set {prop}='{extracted_props[prop]}' from any member")
            else:
                # No values found - use fallback
                extracted_props[prop] = 'Unknown' if prop == 'subregion' else '' 
                print(f"  ⚠️ Using fallback value '{extracted_props[prop]}' for {prop}")
        
        # For meeting time, if not already set for new circles
        if 'meeting_time' not in extracted_props or not extracted_props['meeting_time']:
            # First check for combined day-time values
            if property_values['meeting_day_time']:
                # Prioritize continuing member data for existing circles
                if is_continuing_circle:
                    time_values = [item for item in property_values['meeting_day_time'] if item['is_continuing']]
                    if not time_values:
                        time_values = property_values['meeting_day_time']
                else:
                    # For new circles, just use any value with preference for proposed_NEW_DayTime
                    proposed_values = [item for item in property_values['meeting_day_time'] 
                                     if 'proposed_NEW_DayTime' in item['column']]
                    time_values = proposed_values if proposed_values else property_values['meeting_day_time']
                
                extracted_props['meeting_time'] = time_values[0]['value']
                print(f"  ✅ Set meeting_time='{extracted_props['meeting_time']}' from combined day-time column")
            
            # If no combined data, try to build from separate day and time
            elif len(property_values['meeting_day']) > 0 and len(property_values['meeting_time']) > 0:
                # Prioritize continuing member data for existing circles
                if is_continuing_circle:
                    day_values = [item for item in property_values['meeting_day'] if item['is_continuing']]
                    time_values = [item for item in property_values['meeting_time'] if item['is_continuing']]
                    
                    # If no continuing member data, use any data
                    if not day_values:
                        day_values = property_values['meeting_day']
                    if not time_values:
                        time_values = property_values['meeting_time']
                else:
                    # For new circles, just use any value
                    day_values = property_values['meeting_day']
                    time_values = property_values['meeting_time']
                
                # Format the combined day and time
                day = day_values[0]['value']
                time = time_values[0]['value']
                
                # Standardize time format
                if time.lower() == 'evening':
                    time = 'Evenings'
                elif time.lower() == 'day':
                    time = 'Days'
                    
                formatted_time = f"{day} ({time})"
                extracted_props['meeting_time'] = formatted_time
                print(f"  ✅ Set combined meeting_time='{formatted_time}' from separate day and time columns")
                
            elif len(property_values['meeting_day']) > 0:
                # Just use the day
                if is_continuing_circle:
                    day_values = [item for item in property_values['meeting_day'] if item['is_continuing']]
                    if not day_values:
                        day_values = property_values['meeting_day']
                else:
                    day_values = property_values['meeting_day']
                
                extracted_props['meeting_time'] = day_values[0]['value']
                print(f"  ✅ Set meeting_time='{extracted_props['meeting_time']}' from day only")
                
            elif len(property_values['meeting_time']) > 0:
                # Just use the time
                if is_continuing_circle:
                    time_values = [item for item in property_values['meeting_time'] if item['is_continuing']]
                    if not time_values:
                        time_values = property_values['meeting_time']
                else:
                    time_values = property_values['meeting_time']
                
                extracted_props['meeting_time'] = time_values[0]['value']
                print(f"  ✅ Set meeting_time='{extracted_props['meeting_time']}' from time only")
                
            else:
                # Last resort fallback
                extracted_props['meeting_time'] = 'Unknown'
                print(f"  ⚠️ Using fallback value 'Unknown' for meeting_time")
        
        # Enhanced check for special region circles - always set correct values
        special_region_code = None
        
        # First check for direct circle ID matches - CRITICAL FIX FOR SPECIAL REGIONS
        parts = circle_id.split('-')
        
        # Create a dict to track any incorrect mappings we find
        incorrect_circle_mappings = {
            'IP-MXC-01': 'MXC',  # Mexico City
            'IP-NBO-01': 'NBO',  # Nairobi
            'IP-NAP-01': 'NAP',  # Napa-Sonoma
        }
        
        # Check for direct matches in our problematic circle IDs
        if circle_id in incorrect_circle_mappings:
            special_region_code = incorrect_circle_mappings[circle_id]
            print(f"  🔍 CRITICAL FIX: Directly mapped {circle_id} to special region code {special_region_code}")
        # Otherwise check the usual way
        elif len(parts) >= 3 and parts[0] == 'IP':
            region_code = parts[1]
            if region_code in REGION_CODE_TO_NAME:
                special_region_code = region_code
        
        # IMPROVED: Use the actual region data instead of hardcoded mappings
        if special_region_code is None:
            region_value = extracted_props.get('region', '')
            if isinstance(region_value, str):
                # Check if this is a known region from our mapping
                for code, name in REGION_CODE_TO_NAME.items():
                    if name == region_value:
                        special_region_code = code
                        print(f"  🔍 Found region code '{code}' for region '{region_value}' (circle {circle_id})")
                        break
        
        # For special regions, preserve the original metadata where available
        if special_region_code:
            region_name = REGION_CODE_TO_NAME.get(special_region_code, "Unknown Region")
            print(f"  🔍 Circle {circle_id} is in region: {region_name}")
            
            try:
                # Handle both NEW and existing circle IDs
                if '-NEW-' in circle_id:
                    circle_id_part = circle_id.split('-NEW-')[1]
                else:
                    circle_id_part = circle_id.split('-')[-1]
                
                # Get numeric part of ID for deterministic assignment
                # If not a number, use string hash
                try:
                    circle_num = int(circle_id_part)
                except ValueError:
                    # Use hash of string if not an integer
                    circle_num = sum(ord(c) for c in circle_id)
                
                # IMPROVED: Use actual region values from the data
                region_name = REGION_CODE_TO_NAME.get(special_region_code, "Unknown Region")
                
                # Set region to proper normalized value from our simple mapping
                extracted_props['region'] = region_name
                print(f"  ✅ Set normalized region='{region_name}' for {circle_id}")
                
                # IMPROVED: For problematic circles, use actual participant data for subregion values
                # First check if we have any participants in this circle who have subregion data
                if members_df is not None and not members_df.empty:
                    # Look for Current_Subregion values in the data
                    if 'Current_Subregion' in members_df.columns:
                        # Get non-null subregions from active members
                        valid_subregions = members_df[members_df['Current_Subregion'].notna()]['Current_Subregion'].unique()
                        
                        if len(valid_subregions) > 0:
                            # Use the first available subregion (from actual data)
                            actual_subregion = str(valid_subregions[0])
                            
                            # Log what we're about to do
                            print(f"  ✅ USING ACTUAL DATA: Setting {circle_id} subregion to '{actual_subregion}' from participant data")
                            print(f"  🔍 DETAILS: Previous subregion was '{extracted_props.get('subregion', 'Unknown')}'")
                            
                            # Set the correct subregion from the actual data
                            extracted_props['subregion'] = actual_subregion
                    
                # Different handling for CONTINUING vs NEW circles
                if is_continuing_circle:
                    print(f"  🔄 CONTINUING CIRCLE: {circle_id} - Preserving normalized metadata")
                    
                    # For continuing circles (like NAP, NBO, MXC), get current values for logging
                    current_subregion = extracted_props.get('subregion', 'Unknown')
                    current_meeting_time = extracted_props.get('meeting_time', 'Unknown')
                    
                    # Check if the current subregion needs fixing (missing, unknown, or has Phoenix)
                    needs_subregion_fix = (
                        current_subregion == 'Unknown' or 
                        not current_subregion or 
                        'Phoenix' in current_subregion
                    ) and circle_id not in incorrect_circle_mappings  # Skip if already fixed above
                    
                    # For continuing circles, check if any member has valid subregion info
                    if needs_subregion_fix:
                        continuing_subregions = []
                        for _, row in members_df.iterrows():
                            if 'Status' in row and str(row['Status']).upper() == 'CURRENT-CONTINUING':
                                for col in ['Current_Subregion', 'Requested_Subregion']:
                                    if col in row and not pd.isna(row[col]) and row[col]:
                                        subregion_val = str(row[col])
                                        continuing_subregions.append(subregion_val)
                                        break
                        
                        # Use first valid subregion from continuing members if available
                        if continuing_subregions:
                            extracted_props['subregion'] = continuing_subregions[0]
                            print(f"  ✅ CONTINUING: Using member subregion '{extracted_props['subregion']}' for {circle_id}")
                        else:
                            # Otherwise use first normalized subregion for this region
                            extracted_props['subregion'] = available_subregions[0]
                            print(f"  ✅ CONTINUING: Using primary normalized subregion '{extracted_props['subregion']}' for {circle_id}")
                    else:
                        print(f"  ✅ CONTINUING: Keeping valid subregion '{current_subregion}' for {circle_id}")
                    
                    # Check if meeting time needs fixing (missing or unknown)
                    needs_time_fix = (
                        current_meeting_time == 'Unknown' or 
                        not current_meeting_time
                    )
                    
                    # For continuing circles, check if any member has valid meeting time
                    if needs_time_fix:
                        continuing_meeting_times = []
                        for _, row in members_df.iterrows():
                            if 'Status' in row and str(row['Status']).upper() == 'CURRENT-CONTINUING':
                                for col in ['Current_DayTime', 'Current_Meeting_Day', 'Current Meeting Day']:
                                    if col in row and not pd.isna(row[col]) and row[col]:
                                        time_val = str(row[col])
                                        continuing_meeting_times.append(time_val)
                                        break
                        
                        # Use first valid meeting time from continuing members if available
                        if continuing_meeting_times:
                            from modules.data_processor import standardize_time_preference
                            extracted_props['meeting_time'] = standardize_time_preference(continuing_meeting_times[0])
                            print(f"  ✅ CONTINUING: Using member meeting time '{extracted_props['meeting_time']}' for {circle_id}")
                        else:
                            # Otherwise use deterministic meeting time based on circle ID
                            time_index = circle_num % len(available_meeting_times)
                            extracted_props['meeting_time'] = available_meeting_times[time_index]
                            print(f"  ✅ CONTINUING: Using deterministic meeting time '{extracted_props['meeting_time']}' for {circle_id}")
                    else:
                        print(f"  ✅ CONTINUING: Keeping valid meeting time '{current_meeting_time}' for {circle_id}")
                
                else:
                    # For NEW circles, use deterministic assignment for consistency
                    print(f"  🆕 NEW CIRCLE: {circle_id} - Using deterministic normalized assignment")
                    
                    # For new circles, get current values for logging
                    current_subregion = extracted_props.get('subregion', 'Unknown')
                    current_meeting_time = extracted_props.get('meeting_time', 'Unknown')
                    
                    # For new circles, use deterministic subregion assignment
                    subregion_index = circle_num % len(available_subregions)
                    extracted_props['subregion'] = available_subregions[subregion_index]
                    print(f"  ✅ NEW CIRCLE: Set normalized subregion '{extracted_props['subregion']}' for {circle_id}")
                    
                    # For new circles, set meeting time deterministically if needed
                    if current_meeting_time == 'Unknown' or not current_meeting_time:
                        time_index = circle_num % len(available_meeting_times)
                        extracted_props['meeting_time'] = available_meeting_times[time_index]
                        print(f"  ✅ NEW CIRCLE: Set meeting time '{extracted_props['meeting_time']}' for {circle_id}")
                    else:
                        print(f"  ✅ NEW CIRCLE: Keeping valid meeting time '{current_meeting_time}' for {circle_id}")
                
                # This line is handled at the beginning of the special region code block
                # We already set the region to the normalized value earlier, so this code is redundant
                # But we'll keep a debug message for clarity
                print(f"  ✅ FINAL CHECK: Region for {circle_id} is '{extracted_props.get('region', 'Unknown')}'")
                print(f"  ✅ FINAL CHECK: Subregion for {circle_id} is '{extracted_props.get('subregion', 'Unknown')}'")
                print(f"  ✅ FINAL CHECK: Meeting time for {circle_id} is '{extracted_props.get('meeting_time', 'Unknown')}'")
                
                # Set proposed_NEW fields to ensure they match our normalized values
                # This is critical for continuing circles to have their original metadata preserved throughout the process
                circle_metadata[circle_id]['proposed_NEW_Region'] = extracted_props.get('region', '')
                circle_metadata[circle_id]['proposed_NEW_Subregion'] = extracted_props.get('subregion', '')
                circle_metadata[circle_id]['proposed_NEW_DayTime'] = extracted_props.get('meeting_time', '')
                
                # Special handling for Peninsula and Phoenix confusion
                if special_region_code == 'PSA':
                    # Double-check for any Phoenix references anywhere in metadata
                    if any('Phoenix' in str(val) for val in extracted_props.values()):
                        print(f"  ⚠️ CRITICAL: Found Phoenix reference in Peninsula circle metadata after fixes")
                        
                        # Force-check all fields
                        for key, val in extracted_props.items():
                            if 'Phoenix' in str(val):
                                print(f"  🛠️ Final cleanup: Found Phoenix in '{key}': {val}")
                        
                        # Make sure region is correct
                        extracted_props['region'] = 'Peninsula'
                
            except Exception as e:
                region_name = REGION_CODE_TO_NAME.get(special_region_code, "Unknown")
                print(f"  ⚠️ Error processing {region_name} circle: {str(e)}")
                
                # Use actual participant data for defaults if possible
                if members_df is not None and not members_df.empty:
                    # Try to get region/subregion from members
                    if 'Current_Subregion' in members_df.columns and members_df['Current_Subregion'].notna().any():
                        extracted_props['subregion'] = members_df['Current_Subregion'].dropna().iloc[0]
                    else:
                        extracted_props['subregion'] = "Unknown"
                        
                    if 'Current_Region' in members_df.columns and members_df['Current_Region'].notna().any():
                        extracted_props['region'] = members_df['Current_Region'].dropna().iloc[0]
                    else:
                        extracted_props['region'] = region_name
                        
                    if 'Current_Meeting_Time' in members_df.columns and members_df['Current_Meeting_Time'].notna().any():
                        extracted_props['meeting_time'] = members_df['Current_Meeting_Time'].dropna().iloc[0]
                    else:
                        extracted_props['meeting_time'] = "Unknown"
                else:
                    # No valid member data, use basic defaults
                    extracted_props['subregion'] = "Unknown"
                    extracted_props['meeting_time'] = "Unknown"
                    extracted_props['region'] = region_name
        
        # Set the extracted properties in circle metadata
        circle_metadata[circle_id]['region'] = extracted_props['region']
        circle_metadata[circle_id]['subregion'] = extracted_props['subregion']
        circle_metadata[circle_id]['meeting_time'] = extracted_props['meeting_time']
        
        # Verify and fix member counts
        # Calculate the actual member count based on the members list
        if 'members' in circle_metadata[circle_id] and isinstance(circle_metadata[circle_id]['members'], list):
            real_member_count = len(circle_metadata[circle_id]['members'])
            
            # If member_count doesn't match the real count, update it
            if circle_metadata[circle_id].get('member_count', 0) != real_member_count:
                print(f"  🛠️ FIXING: Circle {circle_id} has incorrect member_count {circle_metadata[circle_id].get('member_count', 0)} but actually has {real_member_count} members")
                circle_metadata[circle_id]['member_count'] = real_member_count
                
            # Make sure member count is never zero for circles with members
            if real_member_count > 0 and circle_metadata[circle_id].get('member_count', 0) == 0:
                print(f"  🛠️ FIXING: Circle {circle_id} has zero member_count but has {real_member_count} members")
                circle_metadata[circle_id]['member_count'] = real_member_count
        
        # Print the final extracted values
        print(f"  📊 FINAL VALUES for circle {circle_id}:")
        print(f"    Region: {circle_metadata[circle_id]['region']}")
        print(f"    Subregion: {circle_metadata[circle_id]['subregion']}")
        print(f"    Meeting Time: {circle_metadata[circle_id]['meeting_time']}")
        
        # Property extraction is now done in the improved code above
                    
        # Check if the circle was in the original circles dataframe
        if circle_id in original_circle_info:
            # Copy properties not already set
            for key, value in original_circle_info[circle_id].items():
                # Safe handling for DataFrame/Series truth value ambiguity
                prop_exists = key in circle_metadata[circle_id]
                if not prop_exists:
                    circle_metadata[circle_id][key] = value
                else:
                    # Use our safe_isna helper function to handle all types
                    val = circle_metadata[circle_id][key]
                    if safe_isna(val):  # This handles both scalar and array-like values safely
                        circle_metadata[circle_id][key] = value
                    
        # Count hosts if host column exists
        if 'host' in results_df.columns:
            always_hosts = len(members_df[members_df['host'] == 'Always'])
            sometimes_hosts = len(members_df[members_df['host'] == 'Sometimes'])
            circle_metadata[circle_id]['always_hosts'] = always_hosts
            circle_metadata[circle_id]['sometimes_hosts'] = sometimes_hosts
                    
        # Add is_existing flag and new_members count
        if 'Status' in results_df.columns:
            # CRITICAL FIX: Safely process Status values that might be floats
            def safe_check_status(status_value, target_status):
                if isinstance(status_value, (int, float)):
                    # Numeric values can't be compared to strings directly
                    return False
                return str(status_value).upper() == target_status
            
            new_members = sum(1 for _, row in members_df.iterrows() 
                             if safe_check_status(row.get('Status', ''), 'NEW'))
            continuing_members = sum(1 for _, row in members_df.iterrows() 
                                    if safe_check_status(row.get('Status', ''), 'CURRENT-CONTINUING'))
            
            circle_metadata[circle_id]['new_members'] = new_members
            circle_metadata[circle_id]['continuing_members'] = continuing_members
            circle_metadata[circle_id]['is_existing'] = continuing_members > 0
            circle_metadata[circle_id]['is_new_circle'] = continuing_members == 0
            
            # Calculate max_additions for continuing circles
            if continuing_members > 0:  # This is an existing circle
                # CRITICAL FIX: Handle status values that might be floats
                # Create a safe function to check status values
                def safe_status_match(row_status, target_status):
                    if isinstance(row_status, (int, float)):
                        return False  # Numeric values never match string statuses
                    return str(row_status).upper() == target_status
                
                # Count unique continuing members and new members for accurate sizing
                unique_continuing = sum(1 for p_id in member_ids_set if any(
                    safe_status_match(status, 'CURRENT-CONTINUING')
                    for status in members_df[members_df[id_column] == p_id]['Status']))
                
                unique_new = sum(1 for p_id in member_ids_set if any(
                    safe_status_match(status, 'NEW')
                    for status in members_df[members_df[id_column] == p_id]['Status']))
                
                print(f"  Circle {circle_id}: {unique_continuing} unique continuing members, {unique_new} unique new members")
                
                # CRITICAL FIX: Update member_count for continuing circles
                # This fixes the issue where continuing circles show 1 member in Circle Composition
                total_members = len(member_ids)
                
                # Set member_count to the actual total of unique members
                circle_metadata[circle_id]['member_count'] = total_members
                print(f"  ✅ FIXED: Set member_count to match total unique members ({total_members}) for circle {circle_id}")
                
                # Get configurable maximum circle size (default to 8 if not set)
                import streamlit as st
                max_circle_size = st.session_state.get('max_circle_size', 8) if 'st' in globals() else 8
                
                # CRITICAL CHECK: Enforce configurable member limit for continuing circles
                # Exception: preserve continuing-only circles that exceed the new limit
                has_new_members = unique_new > 0
                if total_members > max_circle_size and has_new_members:
                    print(f"  ⚠️ WARNING: Circle {circle_id} exceeds maximum size with {total_members} members!")
                    print(f"    This includes {unique_continuing} continuing members and {unique_new} new members")
                    if unique_new > 0:
                        print(f"    This circle should not have accepted new members as it already has {unique_continuing} continuing members")
                
                # First, check if max_additions exists in original data
                if circle_id in original_circle_info and 'max_additions' in original_circle_info[circle_id]:
                    # Use the existing max_additions value from optimization
                    max_additions = original_circle_info[circle_id]['max_additions']
                    
                    # UPDATED FIX: Enforce the configurable member limit
                    # Even if optimizer allowed more, we need to correct it here
                    # Exception: preserve continuing-only circles that exceed the new limit
                    if has_new_members and total_members >= max_circle_size:
                        # Already at or over capacity, force max_additions to 0
                        print(f"  ⚠️ FIXING: Circle {circle_id} is at/over capacity. Setting max_additions to 0 (was {max_additions})")
                        max_additions = 0
                    elif has_new_members and total_members + max_additions > max_circle_size:
                        # Would exceed capacity, adjust max_additions
                        corrected_max = max_circle_size - total_members
                        print(f"  ⚠️ FIXING: Circle {circle_id} would exceed capacity. Adjusting max_additions from {max_additions} to {corrected_max}")
                        max_additions = corrected_max
                    
                    circle_metadata[circle_id]['max_additions'] = max_additions
                    print(f"  Preserved max_additions={max_additions} for circle {circle_id}")
                else:
                    # Calculate max_additions based on continuing circle rules
                    # 1. For continuing circles, never exceed configurable maximum
                    # 2. For small circles (<5 members), add enough to reach 5 regardless of preferences
                    # Exception: preserve continuing-only circles that exceed the new limit
                    
                    if total_members < 5:
                        # Small circle - can add members to reach 5
                        max_additions = 5 - total_members
                        print(f"  Small circle {circle_id}: {total_members} members, calculated max_additions={max_additions}")
                    elif not has_new_members and total_members > max_circle_size:
                        # Continuing-only circle that exceeds new limit - preserve as-is
                        max_additions = 0
                        print(f"  Preserving continuing-only circle {circle_id}: {total_members} members (exceeds max {max_circle_size}), max_additions=0")
                    else:
                        # Regular continuing circle - never exceed configurable maximum
                        max_additions = max(0, max_circle_size - total_members)
                        print(f"  Continuing circle {circle_id}: {total_members} members, calculated max_additions={max_additions}")
                    
                    circle_metadata[circle_id]['max_additions'] = max_additions
            else:
                # New circle - use configurable max size
                total_members = len(member_ids)
                # CRITICAL FIX: For new circles, set consistent max additions
                # Always set max_additions to configurable maximum for new circles
                max_additions = max_circle_size
                
                # CRITICAL FIX: For new circles, member_count should match new_members
                # These circles often show 1 (likely due to member counting issues)
                new_members_count = circle_metadata[circle_id].get('new_members', total_members)
                
                # ALWAYS set member_count to match new_members for new circles
                if new_members_count > 0:
                    # For new circles, we ALWAYS want to set the member count to at least the new members count
                    circle_metadata[circle_id]['member_count'] = max(circle_metadata[circle_id].get('member_count', 0), new_members_count)
                    print(f"  ✅ CRITICAL NEW CIRCLE FIX: Set member_count to {circle_metadata[circle_id]['member_count']} for {circle_id}")
                elif total_members > 1:
                    # If new_members_count is 0 but we have actual members, use that count
                    circle_metadata[circle_id]['member_count'] = total_members
                    circle_metadata[circle_id]['new_members'] = total_members  # Also update new_members to match
                    print(f"  ✅ CRITICAL NEW CIRCLE FIX: Set member_count and new_members to total_members ({total_members}) for {circle_id}")
                
                circle_metadata[circle_id]['max_additions'] = max_additions
                print(f"  New circle {circle_id}: {new_members_count} members, max_additions set to {max_additions}")
            
    # Convert circle metadata to DataFrame
    circles_df = pd.DataFrame(list(circle_metadata.values()))
    
    # Add metadata_source to indicate these circles have enhanced optimizer metadata
    if use_standardized_metadata and not circles_df.empty:
        circles_df['metadata_source'] = 'optimizer'
        print(f"  ✅ Added metadata_source='optimizer' to {len(circles_df)} circles")
    
    # If we have results, do final verification and fix of member counts based on actual members list
    print("\n🔍 CRITICAL VERIFICATION: Double-checking member counts against actual members list")
    
    # Post-process the dataframe
    if not circles_df.empty:
        # CRITICAL FIX: Verify each circle's member count matches its actual members list length
        count_fixed = 0
        for i, row in circles_df.iterrows():
            circle_id = row['circle_id']
            members_list = row['members'] if 'members' in row else []
            
            # Get the actual member count from the members list
            actual_count = len(members_list) if isinstance(members_list, list) else 0
            
            # If member_count doesn't match the actual number of members, fix it
            if 'member_count' in row and row['member_count'] != actual_count and actual_count > 0:
                old_count = row['member_count']
                circles_df.at[i, 'member_count'] = actual_count
                count_fixed += 1
                print(f"  ✅ FIXED: Circle {circle_id} member_count corrected from {old_count} to {actual_count}")
                
                # Special tracking for test circles
                if circle_id in test_circle_ids:
                    print(f"  🔎 TEST CIRCLE {circle_id}: member_count updated to {actual_count}")
                    print(f"    Members: {members_list}")
        
        print(f"  Fixed member counts for {count_fixed} circles")
        
        # Ensure numeric columns are integers
        for col in ['member_count', 'new_members', 'always_hosts', 'sometimes_hosts', 'max_additions']:
            if col in circles_df.columns:
                circles_df[col] = pd.to_numeric(circles_df[col], errors='coerce').fillna(0).astype(int)
                
        # Sort by circle ID
        if 'circle_id' in circles_df.columns:
            circles_df = circles_df.sort_values('circle_id')
    
    print(f"  Successfully created circles DataFrame with {len(circles_df)} circles")
    
    # Critical debug for member counts - we need to make a direct fix
    print("\n🔍 CRITICAL DEBUG: MEMBER COUNT VERIFICATION - Final Fix")
    count_fixed = 0
    new_count_fixed = 0

    # First, create a members count dictionary based on the actual "members" list for each circle
    print("\n💡 MEMBER COUNT RECALCULATION BASED ON MEMBERS LIST")
    member_count_from_list = {}
    for i, row in circles_df.iterrows():
        circle_id = row['circle_id']
        if 'members' in row and isinstance(row['members'], list):
            actual_count = len(row['members'])
            member_count_from_list[circle_id] = actual_count
            
            # Compare with existing count
            current_count = row.get('member_count', 0)
            if current_count != actual_count:
                print(f"  ⚠️ Circle {circle_id} has member_count={current_count} but members list has {actual_count} items")

    # Check all circles for member count issues
    for i, row in circles_df.iterrows():
        circle_id = row['circle_id']
        is_continuing = row.get('is_existing', False)
        is_new = row.get('is_new_circle', False) 
        member_count = row.get('member_count', 0)
        continuing_count = row.get('continuing_members', 0)
        new_count = row.get('new_members', 0)
        
        # CRITICAL FIX FOR ALL CIRCLES: Calculate proper member count from members list
        if circle_id in member_count_from_list:
            actual_count = member_count_from_list[circle_id]
            
            # CRITICAL FIX: For new circles, always use the higher of:
            # 1. Members list count, 2. New members count, or 3. Current member count
            if circle_id.startswith('IP-NEW'):
                # Get the new members count
                new_members_count = row.get('new_members', 0)
                
                # Find the highest count for this new circle
                best_count = max(actual_count, member_count, new_members_count)
                
                # Only update if we have a better count
                if best_count > member_count:
                    old_count = member_count
                    circles_df.at[i, 'member_count'] = best_count
                    new_count_fixed += 1
                    print(f"  🔧 CRITICAL FIX FOR NEW CIRCLE: {circle_id} member_count updated from {old_count} to {best_count}")
                    print(f"    Source counts: members_list={actual_count}, current={member_count}, new_members={new_members_count}")
                # Even if the counts match, always set member_count to new_members for new circles
                elif best_count <= 1 and new_members_count > 1:
                    old_count = member_count
                    circles_df.at[i, 'member_count'] = new_members_count
                    new_count_fixed += 1
                    print(f"  🔧 DIRECT OVERRIDE FOR NEW CIRCLE: {circle_id} member_count forced from {old_count} to {new_members_count}")
        
        # Debug output for identified test circles or any circles with issues
        if circle_id in test_circle_ids or is_continuing or member_count == 1:
            print(f"  🔍 CIRCLE CHECK: {circle_id}:")
            print(f"    member_count = {member_count}, continuing_members = {continuing_count}, new_members = {new_count}")
            print(f"    is_existing = {is_continuing}, is_new_circle = {is_new}")
            if circle_id in member_count_from_list:
                print(f"    members list length = {member_count_from_list[circle_id]}")
            
            # Check if we have a mismatch - existing circle with only 1 member
            if is_continuing and continuing_count > 0 and member_count == 1:
                # CRITICAL FIX: Based on continuing_members, force update the member_count
                old_count = member_count
                
                # Use the higher of continuing_count or the actual members list length
                correct_count = continuing_count
                if circle_id in member_count_from_list and member_count_from_list[circle_id] > correct_count:
                    correct_count = member_count_from_list[circle_id]
                    
                circles_df.at[i, 'member_count'] = correct_count
                count_fixed += 1
                print(f"  🔧 DIRECT FIX: Circle {circle_id} member_count forced from {old_count} to {correct_count}")
            
            # For continuing circles with incorrect counts, make sure they show the right number
            elif is_continuing and member_count != (continuing_count + new_count) and (continuing_count + new_count) > 0:
                old_count = member_count
                correct_count = continuing_count + new_count
                
                # Verify against members list if available
                if circle_id in member_count_from_list and member_count_from_list[circle_id] > correct_count:
                    correct_count = member_count_from_list[circle_id]
                
                circles_df.at[i, 'member_count'] = correct_count
                count_fixed += 1
                print(f"  🔧 DIRECT FIX: Circle {circle_id} member_count corrected from {old_count} to {correct_count}")
    
    if count_fixed > 0:
        print(f"  ✅ Direct-fixed member counts for {count_fixed} continuing circles with incorrect counts")
    
    if new_count_fixed > 0:
        print(f"  ✅ Direct-fixed member counts for {new_count_fixed} new circles with incorrect counts")

    # FINAL FORCED CHECK FOR NEW CIRCLES: This is our last check to ensure all new circles
    # have the correct member count
    print("\n🔍 FINAL CHECK: Verifying all NEW circles have member count matching new member count")
    final_fixed = 0
    for i, row in circles_df.iterrows():
        circle_id = row['circle_id']
        
        if circle_id.startswith('IP-NEW'):
            member_count = row.get('member_count', 0)
            new_members_count = row.get('new_members', 0)
            
            # For new circles, member_count should match new_members if new_members > 0
            if new_members_count > 0 and member_count != new_members_count:
                old_count = member_count
                circles_df.at[i, 'member_count'] = new_members_count
                final_fixed += 1
                print(f"  🚨 FINAL FIX: NEW circle {circle_id} member_count forced from {old_count} to {new_members_count}")
            elif member_count == 1 and 'members' in row and isinstance(row['members'], list) and len(row['members']) > 1:
                # If we still have member_count=1 but actual members list is larger, use that
                actual_count = len(row['members'])
                old_count = member_count
                circles_df.at[i, 'member_count'] = actual_count
                final_fixed += 1
                print(f"  🚨 FINAL MEMBERS LIST FIX: NEW circle {circle_id} member_count forced from {old_count} to {actual_count}")
    
    if final_fixed > 0:
        print(f"  ✅ Final-fixed member counts for {final_fixed} new circles with forced correction")
    
    # Note: We leave meeting times blank if not available, per user instruction
    missing_time_count = 0
    for i, row in circles_df.iterrows():
        circle_id = row['circle_id']
        
        # Count how many meeting times are missing but don't add fallbacks
        if 'meeting_time' not in row or safe_isna(row['meeting_time']):
            missing_time_count += 1
            print(f"  ⚠️ Circle {circle_id} has missing meeting time - leaving blank")
            
    if missing_time_count > 0:
        print(f"  Found {missing_time_count} circles with missing meeting times - leaving blank as instructed")
    
    # Enhanced debugging - show all tracked test circles
    for circle_id in test_circle_ids:
        # Dynamic column detection to handle different DataFrame structures
        circle_id_column = None
        for col in circles_df.columns:
            if 'circle' in col.lower() and 'id' in col.lower():
                circle_id_column = col
                break
        
        if circle_id_column:
            test_circle = circles_df[circles_df[circle_id_column] == circle_id]
        else:
            # Fallback: look for any column containing the circle_id value
            test_circle = pd.DataFrame()  # Empty DataFrame if no matching column found
            for col in circles_df.columns:
                if circles_df[col].astype(str).eq(circle_id).any():
                    test_circle = circles_df[circles_df[col] == circle_id]
                    break
        if not test_circle.empty:
            row = test_circle.iloc[0]
            print(f"  🔍 FINAL TEST CIRCLE: {circle_id}: {row['member_count']} members "
                 f"({row.get('new_members', 0)} new, {row.get('continuing_members', 0)} continuing, "
                 f"max_additions={row.get('max_additions', 0)})")
            print(f"     Meeting time: '{row.get('meeting_time', 'None')}'")
        else:
            print(f"  ⚠️ TEST CIRCLE {circle_id} not found in final circles DataFrame")
    
    # Debug - show a few sample circles
    if not circles_df.empty and len(circles_df) > 0:
        print("  Sample circles before renumbering:")
        for i, (_, row) in enumerate(circles_df.head(5).iterrows()):
            print(f"    {i+1}. {row['circle_id']}: {row['member_count']} members "
                 f"({row.get('new_members', 0)} new, {row.get('continuing_members', 0)} continuing, "
                 f"max_additions={row.get('max_additions', 0)})")
    
    # Apply virtual circle renumbering first (for GMT offset removal)
    circles_df = renumber_virtual_circles_by_gmt_offset(circles_df)
    
    # Apply sequential renumbering to ensure consistent circle IDs
    circles_df = renumber_circles_sequentially(circles_df)
    
    # Debug - show sample circles after renumbering
    if not circles_df.empty and len(circles_df) > 0:
        print("  Sample circles after renumbering:")
        for i, (_, row) in enumerate(circles_df.head(5).iterrows()):
            print(f"    {i+1}. {row['circle_id']}: {row['member_count']} members "
                 f"({row.get('new_members', 0)} new, {row.get('continuing_members', 0)} continuing, "
                 f"max_additions={row.get('max_additions', 0)})")
    
    # Add a validation step to verify our metadata is correct
    print("\n🔍 VALIDATING CIRCLE METADATA: Checking for consistent metadata")
    metadata_source_count = 0
    if 'metadata_source' in circles_df.columns:
        metadata_source_count = (circles_df['metadata_source'] == 'optimizer').sum()
        print(f"  Found {metadata_source_count}/{len(circles_df)} circles with optimizer metadata")
    else:
        print("  ⚠️ No metadata_source column found - standardized metadata may not be in use")
    
    # Check for any member_count=0 issues
    zero_member_count = len(circles_df[circles_df['member_count'] == 0]) if 'member_count' in circles_df.columns else 0
    if zero_member_count > 0:
        print(f"  ⚠️ Found {zero_member_count} circles with member_count=0")
    else:
        print("  ✅ No circles with member_count=0 detected")
    
    # ENHANCED APPROACH: Use CircleMetadataManager to store circles data
    print("\n🔄 INTEGRATING WITH METADATA MANAGER: Creating centralized circle data store")
    try:
        # Import the CircleMetadataManager
        from utils.circle_metadata_manager import CircleMetadataManager
        from utils.feature_flags import get_flag
        
        # Check if we're using standardized metadata
        use_standardized_metadata = get_flag('use_optimizer_metadata')
        
        # Convert circles_df to list of dictionaries for the manager
        if hasattr(circles_df, 'to_dict'):
            circle_list = circles_df.to_dict('records')
            print(f"  Converted DataFrame with {len(circles_df)} circles to list format")
            
            # If results was passed as a parameter and is available, use it
            results_df = None
            if isinstance(results, pd.DataFrame):
                results_df = results
                print(f"  Using results DataFrame with {len(results_df)} participants")
            
            # Create metadata manager instance for use in other components
            manager = CircleMetadataManager().initialize_from_optimizer(circle_list, results_df)
            print(f"  ✅ Successfully created CircleMetadataManager with {len(circle_list)} circles")
            
            # Add enhanced debugging information
            if use_standardized_metadata:
                # Check if all circles in the metadata manager are properly tracked
                all_manager_circles = manager.get_all_circles()
                print(f"  💡 CircleMetadataManager contains {len(all_manager_circles)} circles")
                
                if len(all_manager_circles) > 0:
                    # Show a few sample circles with their key metadata
                    print("  Sample circles in metadata manager:")
                    for i, circle in enumerate(all_manager_circles[:3]):
                        print(f"    {i+1}. {circle['circle_id']}: {circle.get('member_count', 'Unknown')} members, "
                              f"metadata_source={circle.get('metadata_source', 'Unknown')}")
            
            # Apply final normalization to all subregion values in the DataFrame
            if not circles_df.empty and 'subregion' in circles_df.columns:
                print("\n🔄 FINAL NORMALIZATION: Ensuring all subregion values are normalized")
                # Store original values for logging
                original_values = circles_df['subregion'].copy()
                
                # Apply normalization to all subregion values
                circles_df['subregion'] = circles_df['subregion'].apply(normalize_subregion)
                
                # Count how many values were normalized
                normalized_count = (original_values != circles_df['subregion']).sum()
                if normalized_count > 0:
                    print(f"  ✅ Normalized {normalized_count} subregion values in final DataFrame")
                else:
                    print("  ✓ All subregion values were already normalized")
            
            # This manager object will be available for use by caller, but we still return the DataFrame
            # for backward compatibility with existing code
            return circles_df
        else:
            print("  ⚠️ Failed to convert circles_df to dictionary format - not a DataFrame?")
    except Exception as e:
        print(f"  ⚠️ Error creating CircleMetadataManager: {str(e)}")
        print("  Continuing with DataFrame return only")
    
    # Apply final normalization to all subregion values in the DataFrame
    if not circles_df.empty and 'subregion' in circles_df.columns:
        print("\n🔄 FINAL NORMALIZATION: Ensuring all subregion values are normalized")
        # Store original values for logging
        original_values = circles_df['subregion'].copy()
        
        # Apply normalization to all subregion values
        circles_df['subregion'] = circles_df['subregion'].apply(normalize_subregion)
        
        # Count how many values were normalized
        normalized_count = (original_values != circles_df['subregion']).sum()
        if normalized_count > 0:
            print(f"  ✅ Normalized {normalized_count} subregion values in final DataFrame")
        else:
            print("  ✓ All subregion values were already normalized")
    
    return circles_df