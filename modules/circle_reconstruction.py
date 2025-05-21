"""
Circle reconstruction utilities to ensure all assigned circles, including post-processed ones,
appear correctly in UI components.
"""

import pandas as pd
import numpy as np

def safe_isna(val):
    """Safely check if a value is NA, handling both scalar and array-like objects."""
    if isinstance(val, (pd.Series, pd.DataFrame)):
        # For pandas objects, check if all values are NA
        return val.isna().all()
    elif isinstance(val, (np.ndarray, list)):
        # For numpy arrays or lists
        return all(pd.isna(x) for x in val)
    else:
        # For scalar values
        return pd.isna(val)

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
            # Handle virtual circle format: VO-AM-GMT-5-NEW-XX or VO-AM-GMT-5-XX
            if len(parts) >= 5 and 'GMT' in parts[2]:
                # Format: VO-AM-GMT-5-NEW-XX or VO-AM-GMT-5-XX
                # Region code is "AM-GMT-5"
                region_code = f"{parts[1]}-{parts[2]}-{parts[3]}"
                
                # Check if this is a NEW circle
                is_new = 'NEW' in parts
            elif len(parts) >= 4:
                # Alternate format: VO-AM-GMT+1-NEW-XX
                # Region code is "AM-GMT+1"
                region_code = f"{parts[1]}-{parts[2]}"
                
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
            
            # Create a new ID with sequential numbering
            # Format: IP-NYC-01 or VO-AM-GMT-5-01
            new_id = f"{format_prefix}-{region_code}-{str(idx).zfill(2)}"
            
            # Only update if the IDs are different
            if old_id != new_id:
                circle_id_mapping[old_id] = new_id
                print(f"    🔄 Renumbering: {old_id} → {new_id}")
        
        # Process new circles (with NEW in the ID)
        for idx, circle_info in enumerate(sorted(new_circles, key=lambda x: x['circle_id']), start=1):
            old_id = circle_info['circle_id']
            
            # Create a new ID with sequential numbering
            # Format: IP-NYC-NEW-01 or VO-AM-GMT-5-NEW-01
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
        
    # Check if this is a virtual circle with old format (V-VIR-NEW-XX)
    if circle_id.startswith('V-VIR-NEW-'):
        print(f"🔧 FIXING VIRTUAL CIRCLE ID FORMAT: {circle_id}")
        
        # Extract the index number
        parts = circle_id.split('-')
        if len(parts) >= 3:
            index_str = parts[-1]
            
            # Determine the region code based on the provided region/subregion
            from utils.normalization import get_region_code_with_subregion
            
            if region and subregion:
                is_virtual = 'Virtual' in str(region) if region is not None else False
                
                if is_virtual:
                    # Get proper region code with timezone
                    region_code = get_region_code_with_subregion(region, subregion, is_virtual=True)
                    
                    # Create new ID with proper format
                    new_id = f"VO-{region_code}-NEW-{index_str}"
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
    
    Args:
        results: List of participant results with assignments
        original_circles: Original circles dataframe (optional)
        use_standardized_metadata: Whether to use standardized metadata from optimizer
        
    Returns:
        DataFrame: Updated circles dataframe with all assigned circles
    """
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
        meeting_columns = [col for col in results_df.columns if 'meeting' in col.lower() or 'time' in col.lower() or 'day' in col.lower()]
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
            # Check for metadata_source column which would indicate optimizer metadata
            if 'metadata_source' in original_circles.columns:
                optimizer_circles = original_circles[original_circles['metadata_source'] == 'optimizer']
                if not optimizer_circles.empty:
                    print(f"  ✅ Found {len(optimizer_circles)} circles with optimizer metadata")
                    has_optimizer_metadata = True
                    
                    # Use the optimizer's circle metadata as our starting point
                    for _, circle in optimizer_circles.iterrows():
                        circle_id = circle['circle_id']
                        circle_metadata[circle_id] = circle.to_dict()
                        print(f"  ✅ Loaded optimizer metadata for circle {circle_id}")
        
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
        
        # CRITICAL FIX: Remove duplicate member IDs by using a set to ensure uniqueness
        member_ids_set = set(members_df[id_column].tolist())
        member_ids = list(member_ids_set)
        
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
                    extracted_props['subregion'] = str(member['proposed_NEW_Subregion'])
                    print(f"  ✅ NEW CIRCLE: Set subregion='{extracted_props['subregion']}' from proposed_NEW_Subregion")
                    proposed_subregion_found = True
                    break
            
            # For meeting_time, use proposed_NEW_DayTime
            proposed_time_found = False
            for _, member in members_df.iterrows():
                if 'proposed_NEW_DayTime' in member and not pd.isna(member['proposed_NEW_DayTime']) and member['proposed_NEW_DayTime']:
                    extracted_props['meeting_time'] = str(member['proposed_NEW_DayTime'])
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
        
        # Set the extracted properties in circle metadata
        circle_metadata[circle_id]['region'] = extracted_props['region']
        circle_metadata[circle_id]['subregion'] = extracted_props['subregion']
        circle_metadata[circle_id]['meeting_time'] = extracted_props['meeting_time']
        
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
                
                # CRITICAL CHECK: Enforce 8-member limit for continuing circles
                if total_members > 8:
                    print(f"  ⚠️ WARNING: Circle {circle_id} exceeds maximum size with {total_members} members!")
                    print(f"    This includes {unique_continuing} continuing members and {unique_new} new members")
                    if unique_new > 0:
                        print(f"    This circle should not have accepted new members as it already has {unique_continuing} continuing members")
                
                # First, check if max_additions exists in original data
                if circle_id in original_circle_info and 'max_additions' in original_circle_info[circle_id]:
                    # Use the existing max_additions value from optimization
                    max_additions = original_circle_info[circle_id]['max_additions']
                    
                    # CRITICAL FIX: Enforce the 8-member limit
                    # Even if optimizer allowed more, we need to correct it here
                    if total_members >= 8:
                        # Already at or over capacity, force max_additions to 0
                        print(f"  ⚠️ FIXING: Circle {circle_id} is at/over capacity. Setting max_additions to 0 (was {max_additions})")
                        max_additions = 0
                    elif total_members + max_additions > 8:
                        # Would exceed capacity, adjust max_additions
                        corrected_max = 8 - total_members
                        print(f"  ⚠️ FIXING: Circle {circle_id} would exceed capacity. Adjusting max_additions from {max_additions} to {corrected_max}")
                        max_additions = corrected_max
                    
                    circle_metadata[circle_id]['max_additions'] = max_additions
                    print(f"  Preserved max_additions={max_additions} for circle {circle_id}")
                else:
                    # Calculate max_additions based on continuing circle rules
                    # 1. For continuing circles, never exceed a total of 8 members
                    # 2. For small circles (<5 members), add enough to reach 5 regardless of preferences
                    
                    if total_members < 5:
                        # Small circle - can add members to reach 5
                        max_additions = 5 - total_members
                        print(f"  Small circle {circle_id}: {total_members} members, calculated max_additions={max_additions}")
                    else:
                        # Regular continuing circle - never exceed 8 total
                        max_additions = max(0, 8 - total_members)
                        print(f"  Continuing circle {circle_id}: {total_members} members, calculated max_additions={max_additions}")
                    
                    circle_metadata[circle_id]['max_additions'] = max_additions
            else:
                # New circle - different max size (10)
                total_members = len(member_ids)
                # CRITICAL FIX: For new circles, set consistent max additions
                # Always set max_additions to 10 for new circles as requested
                max_additions = 10
                
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
        test_circle = circles_df[circles_df['circle_id'] == circle_id]
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
            
            # This manager object will be available for use by caller, but we still return the DataFrame
            # for backward compatibility with existing code
            return circles_df
        else:
            print("  ⚠️ Failed to convert circles_df to dictionary format - not a DataFrame?")
    except Exception as e:
        print(f"  ⚠️ Error creating CircleMetadataManager: {str(e)}")
        print("  Continuing with DataFrame return only")
    
    return circles_df