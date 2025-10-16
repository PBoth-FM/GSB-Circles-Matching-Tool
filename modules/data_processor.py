import pandas as pd
import numpy as np
from utils.normalization import normalize_regions, normalize_subregions
import re
import logging
from utils.data_standardization import normalize_host_status, normalize_member_list, normalize_encoded_id, print_normalization_logs
from utils.feature_flags import get_flag

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_co_leader_max_new_members(value):
    """
    Process co-leader max new members value
    
    Args:
        value: The co-leader max new members value to process
        
    Returns:
        Processed value (always a string for consistent data typing)
    """
    # Handle None/NaN values
    if pd.isna(value) or value is None or value == '':
        return "None"  # Return string "None" for consistency
    
    # Handle 'None' literal string (case-insensitive)
    if isinstance(value, str) and value.strip().lower() == 'none':
        return "None"  # Standardize to "None" string
    
    # Try to convert to integer
    try:
        int_value = int(float(value) if isinstance(value, str) and '.' in value else value)
        # Handle 0 as equivalent to 'None' - no new members should be added
        if int_value == 0:
            return "None"
        # Return as string representation of integer for consistent typing
        return str(int_value)
    except (ValueError, TypeError):
        # If not a valid number, return "None" string
        return "None"

def calculate_class_vintage(gsb_class):
    """
    Calculate the Class Vintage based on GSB Class year
    
    Args:
        gsb_class: The GSB Class year (could be string, float, or int)
        
    Returns:
        Class Vintage category (e.g. "01-10 yrs", "11-20 yrs", etc.)
    """
    # Current year for calculation (as specified)
    current_year = 2025
    
    # Handle missing values
    if pd.isna(gsb_class):
        logging.info(f"Skipping class vintage calculation for NaN value")
        return None
    
    # Convert to string and clean
    gsb_class_str = str(gsb_class).strip()
    
    # Skip empty strings
    if not gsb_class_str:
        logging.info(f"Skipping class vintage calculation for empty value")
        return None
    
    # Log the input value for debugging
    logging.info(f"Calculating Class Vintage for: '{gsb_class}' (type: {type(gsb_class).__name__})")
    
    # Try to extract a 4-digit year from the string using regex
    year_match = re.search(r'(19\d{2}|20\d{2})', gsb_class_str)
    if year_match:
        try:
            class_year = int(year_match.group(1))
            logging.info(f"Extracted year {class_year} from '{gsb_class_str}'")
        except ValueError:
            logging.error(f"Could not convert extracted year '{year_match.group(1)}' to integer")
            return None
    else:
        # If no 4-digit year pattern, try to convert the entire string
        try:
            # For numeric values like 2001.0, convert to integer
            if isinstance(gsb_class, (int, float)):
                class_year = int(gsb_class)
                logging.info(f"Direct conversion of numeric value {gsb_class} to year {class_year}")
            elif gsb_class_str.replace('.', '', 1).isdigit():
                class_year = int(float(gsb_class_str))
                logging.info(f"Converted numeric string {gsb_class_str} to year {class_year}")
            # For other strings that might be numeric
            elif gsb_class_str.isdigit():
                class_year = int(gsb_class_str)
                logging.info(f"Converted digit string '{gsb_class_str}' to year {class_year}")
            else:
                logging.warning(f"Could not extract year from '{gsb_class_str}' - not a valid year format")
                return None
        except (ValueError, TypeError) as e:
            logging.error(f"Error converting '{gsb_class_str}' to year: {str(e)}")
            return None
    
    # Validate year is in reasonable range
    if class_year < 1900 or class_year > 2030:
        logging.warning(f"Year {class_year} is outside reasonable range (1900-2030)")
        return None
    
    # Calculate years since graduation
    years_since = current_year - class_year
    logging.info(f"For class year {class_year}, calculated {years_since} years since graduation")
    
    # Determine decade range
    if years_since <= 10:
        return "01-10 yrs"
    elif years_since <= 20:
        return "11-20 yrs"
    elif years_since <= 30:
        return "21-30 yrs"
    elif years_since <= 40:
        return "31-40 yrs"
    elif years_since <= 50:
        return "41-50 yrs"
    elif years_since <= 60:
        return "51-60 yrs"
    else:
        return "61+ yrs"

def process_data(df, debug_mode=False):
    """
    Process and clean the participant data
    
    Args:
        df: Pandas DataFrame with participant data
        debug_mode: Boolean to enable debug prints
        
    Returns:
        Processed DataFrame
    """
    # Enable special debugging for Peninsula region issues
    DEBUG_PENINSULA_REGION = True
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert column names to standardized format
    processed_df.columns = [col.strip() for col in processed_df.columns]
    
    # Fill NaN values appropriately - using a safer approach to handle missing columns
    for column in ['first_choice_location', 'first_choice_time', 
                   'second_choice_location', 'second_choice_time',
                   'third_choice_location', 'third_choice_time', 'host']:
        # Check if the column exists in the DataFrame
        if column in processed_df.columns:
            # Fill NaN values with empty string
            processed_df[column] = processed_df[column].fillna('')
        else:
            # Add the column with default empty strings if it doesn't exist
            processed_df[column] = ''
    
    # Convert Encoded ID to string if it exists
    if 'Encoded ID' in processed_df.columns:
        processed_df['Encoded ID'] = processed_df['Encoded ID'].apply(normalize_encoded_id)
    
    # Process co-leader max new members if the column exists
    if 'co_leader_max_new_members' in processed_df.columns:
        processed_df['co_leader_max_new_members'] = processed_df['co_leader_max_new_members'].apply(
            process_co_leader_max_new_members
        )
    else:
        # Add the column with default None values if it doesn't exist
        processed_df['co_leader_max_new_members'] = None
        
    # Add standardized host status - find host column with case-insensitive search
    host_col = None
    for col in processed_df.columns:
        if 'host' in col.lower():
            host_col = col
            break
            
    if host_col:
        # Only perform standardization if feature flag is enabled or in debug mode
        if get_flag('use_standardized_host_status') or debug_mode:
            processed_df['host_status_standardized'] = processed_df[host_col].apply(normalize_host_status)
            logging.info(f"Added standardized host status based on '{host_col}'")
            
            # Print host normalization logs if debug mode is enabled
            if debug_mode:
                print("\n🔍 HOST STATUS STANDARDIZATION:")
                # Show sample of original vs standardized values
                sample_size = min(10, len(processed_df))
                sample = processed_df.sample(sample_size) if len(processed_df) > 0 else processed_df
                
                print(f"Sample of {sample_size} standardized host statuses:")
                for _, row in sample.iterrows():
                    original = row[host_col]
                    standardized = row['host_status_standardized']
                    print(f"  '{original}' → '{standardized}'")
                
                # Print full log if requested
                if get_flag('debug_data_standardization'):
                    print_normalization_logs()
    else:
        logging.warning("No host column found to create standardized host status")
        # Create empty standardized column anyway for consistent schema
        processed_df['host_status_standardized'] = 'NEVER'
    
    # Calculate Class Vintage if GSB Class column exists
    # Improved search for GSB Class column with better logging
    gsb_class_columns = []
    
    # Log all available columns for debugging
    logging.info("Searching for GSB Class Year column among these columns:")
    for col in processed_df.columns:
        logging.info(f"  - '{col}'")
        
    # First, try to find the exact column name "GSB Class Year"
    if "GSB Class Year" in processed_df.columns:
        gsb_class_columns.append("GSB Class Year")
        logging.info("Found exact column match: 'GSB Class Year'")
    else:
        # Look for partial matches or variations
        for col in processed_df.columns:
            # Look for 'gsb class' or 'gsbclass' in lowercase version of column name
            if any(term in col.lower().replace(" ", "") for term in ['gsbclass', 'gsb class']):
                gsb_class_columns.append(col)
                logging.info(f"Found GSB Class column via pattern match: '{col}'")

    if gsb_class_columns:
        gsb_class_col = gsb_class_columns[0]
        # Add direct debug output before calculation
        logging.info(f"Using '{gsb_class_col}' for Class Vintage calculation")
        
        # Show sample values from this column
        sample_values = processed_df[gsb_class_col].dropna().head(10).tolist()
        logging.info(f"Sample GSB Class values: {sample_values}")
        
        # Create a safer version of the calculation with more error handling
        try:
            # Convert to numeric first, handling errors
            processed_df['GSB_Class_Numeric'] = pd.to_numeric(
                processed_df[gsb_class_col], 
                errors='coerce'  # Convert errors to NaN instead of raising an exception
            )
            
            # Check if conversion was successful
            valid_count = processed_df['GSB_Class_Numeric'].notna().sum()
            total_count = len(processed_df)
            logging.info(f"Successfully converted {valid_count} out of {total_count} GSB Class values to numeric")
            
            # Apply class vintage calculation to the numeric column
            processed_df['Class_Vintage'] = processed_df['GSB_Class_Numeric'].apply(calculate_class_vintage)
            
            # Debug output
            vintage_counts = processed_df['Class_Vintage'].value_counts()
            vintage_null_count = processed_df['Class_Vintage'].isna().sum()
            logging.info(f"Class Vintage distribution: {vintage_counts}")
            logging.info(f"Null Class Vintage values: {vintage_null_count}")
        except Exception as e:
            logging.error(f"Error calculating Class Vintage: {str(e)}")
            # Add empty Class_Vintage column as fallback
            processed_df['Class_Vintage'] = None
    else:
        # Add empty Class_Vintage column if GSB Class doesn't exist
        logging.error("No GSB Class column found - cannot calculate Class Vintage")
        processed_df['Class_Vintage'] = None
    
    # Filter out participants with "MOVING OUT" status
    processed_df = processed_df[processed_df.get('Status', '') != 'MOVING OUT']
    
    # Process time preferences to standardized format
    for time_col in ['first_choice_time', 'second_choice_time', 'third_choice_time']:
        if time_col in processed_df.columns:
            processed_df[time_col] = processed_df[time_col].apply(standardize_time_preference)
    
    return processed_df

def normalize_data(df, debug_mode=False):
    """
    Normalize region and subregion data
    
    Args:
        df: Pandas DataFrame with processed participant data
        debug_mode: Boolean to enable debug prints
        
    Returns:
        DataFrame with normalized region and subregion data
    """
    # Enable debugging for Peninsula region to catch and fix issues
    DEBUG_PENINSULA_REGION = True
    if df is None or not isinstance(df, pd.DataFrame):
        # Handle the case where df might be a string or None
        return df
        
    normalized_df = df.copy()
    
    # Normalize regions
    for region_col in ['Requested_Region', 'Current_Region', 'Region']:
        if region_col in normalized_df.columns:
            normalized_df[region_col] = normalized_df[region_col].apply(
                lambda x: normalize_regions(x) if pd.notna(x) else x
            )
    
    # Normalize subregions with region context
    for location_col in ['first_choice_location', 'second_choice_location', 'third_choice_location', 'Current_Subregion']:
        if location_col in normalized_df.columns:
            # Get the corresponding region column for context
            region_col = 'Current_Region' if location_col == 'Current_Subregion' else 'Requested_Region'
            
            # Apply normalization with region context
            if region_col in normalized_df.columns:
                # Use both region and subregion for better context-aware normalization
                normalized_df[location_col] = normalized_df.apply(
                    lambda row: normalize_subregions(
                        row[location_col], 
                        region=row.get(region_col)
                    ) if pd.notna(row[location_col]) and row[location_col] != '' else row[location_col],
                    axis=1
                )
            else:
                # Fallback if region column not available
                normalized_df[location_col] = normalized_df[location_col].apply(
                    lambda x: normalize_subregions(x) if pd.notna(x) and x != '' else x
                )
            
    # Add region code detection for virtual regions based on subregion
    if debug_mode:
        print("🔍 Enhanced region normalization with virtual-aware region codes")
    
    # Detect virtual circles based on region
    try:
        if 'Current_Region' in normalized_df.columns and 'Current_Subregion' in normalized_df.columns:
            # Import our enhanced region code function that handles virtual circles
            from utils.normalization import get_region_code_with_subregion
            
            # Create a new column for region codes that takes subregions into account for virtual circles
            normalized_df['Region_Code'] = normalized_df.apply(
                lambda row: get_region_code_with_subregion(
                    row['Current_Region'],
                    row['Current_Subregion'],
                    'Virtual' in str(row['Current_Region']) if pd.notna(row['Current_Region']) else False
                ) if pd.notna(row['Current_Region']) and pd.notna(row['Current_Subregion']) else None,
                axis=1
            )
            
            if debug_mode:
                # Show sample of virtual regions and their codes
                virtual_sample = normalized_df[normalized_df['Current_Region'].str.contains('Virtual', na=False)].head(5) if 'Current_Region' in normalized_df.columns else None
                if virtual_sample is not None and not virtual_sample.empty:
                    print("📊 Sample of virtual region codes:")
                    for _, row in virtual_sample.iterrows():
                        print(f"  Region: {row['Current_Region']}, Subregion: {row['Current_Subregion']}, Code: {row.get('Region_Code', 'Unknown')}")
    except Exception as e:
        print(f"⚠️ WARNING: Error while processing virtual region codes: {str(e)}")
    
    # CRITICAL FIX: Handle "Current-MOVING INTO Region" participants
    # When Raw_Status is "Current-MOVING INTO Region" (any capitalization) and first_choice_location is blank,
    # use Current_Subregion as first_choice_location for the matching process
    if 'Raw_Status' in normalized_df.columns and 'Current_Subregion' in normalized_df.columns:
        # Use case-insensitive regex to match any variation of "MOVING INTO Region"
        moving_into_mask = normalized_df['Raw_Status'].astype(str).str.contains(
            r'moving.*into.*region', 
            case=False, 
            regex=True, 
            na=False
        )
        
        # Only apply when first_choice_location is blank
        blank_first_choice_mask = (
            (normalized_df['first_choice_location'].isna()) | 
            (normalized_df['first_choice_location'] == '')
        )
        
        # Only apply when Current_Subregion exists
        has_subregion_mask = (
            normalized_df['Current_Subregion'].notna() & 
            (normalized_df['Current_Subregion'] != '')
        )
        
        # Combine all conditions
        final_mask = moving_into_mask & blank_first_choice_mask & has_subregion_mask
        
        if final_mask.any():
            # Set first_choice_location to Current_Subregion for these participants
            normalized_df.loc[final_mask, 'first_choice_location'] = normalized_df.loc[final_mask, 'Current_Subregion']
            
            if debug_mode:
                affected_count = final_mask.sum()
                print(f"\n🔄 MOVING INTO Region preprocessing:")
                print(f"  Auto-filled first_choice_location for {affected_count} participants")
                print(f"  Using Current_Subregion as first_choice_location")
    
    # Handle small regions with no subregions
    # For NEW participants from small regions who didn't specify a first_choice_location,
    # set their first_choice_location to their Requested_Region
    small_regions = [
        'San Diego', 'Marin County', 'Nairobi', 'Sao Paulo', 
        'Mexico City', 'Singapore', 'Shanghai', 'Napa/Sonoma', 'Atlanta'
    ]
    
    # Only apply this to non-CURRENT-CONTINUING participants 
    mask = (
        normalized_df['Requested_Region'].isin(small_regions) &
        ((normalized_df['first_choice_location'].isna()) | (normalized_df['first_choice_location'] == '')) &
        (normalized_df['Status'] != 'CURRENT-CONTINUING')
    )
    
    # Set first_choice_location to match Requested_Region for these participants
    normalized_df.loc[mask, 'first_choice_location'] = normalized_df.loc[mask, 'Requested_Region']
    
    # Per PRD 4.3.2: For CURRENT-CONTINUING participants, use current_region; for all others, use requested_region_from_form
    if 'Status' in normalized_df.columns and 'Current_Region' in normalized_df.columns and 'Requested_Region' in normalized_df.columns:
        # Create a derived region column for grouping
        # Per PRD: For CURRENT-CONTINUING participants, use current_region
        # But for "Current-MOVING INTO REGION" and NEW participants, use requested_region
        normalized_df['Derived_Region'] = normalized_df.apply(
            lambda row: row['Requested_Region'] 
                       if row['Status'] == 'NEW' or 
                          (pd.notna(row.get('Raw_Status')) and 'MOVING INTO REGION' in str(row.get('Raw_Status', '')))
                       else row['Current_Region'], 
            axis=1
        )
        
        # Add debug to track region assignments - use the function parameter
        if debug_mode:
            print("\n🔍 CHECKING DERIVED REGION ASSIGNMENTS")
            moving_participants = normalized_df[normalized_df.apply(
                lambda row: pd.notna(row.get('Raw_Status')) and 'MOVING INTO REGION' in str(row.get('Raw_Status', '')),
                axis=1
            )]
            
            if not moving_participants.empty:
                print(f"Found {len(moving_participants)} participants with 'MOVING INTO REGION' status")
                for _, row in moving_participants.iterrows():
                    print(f"ID: {row['Encoded ID']} - Current: {row['Current_Region']} → Requested: {row['Requested_Region']} → Derived: {row['Derived_Region']}")
                    
            # Check test participants specifically
            test_participants = ['73177784103', '50625303450', '72549701782']
            for p_id in test_participants:
                if p_id in normalized_df['Encoded ID'].values:
                    p_row = normalized_df[normalized_df['Encoded ID'] == p_id].iloc[0]
                    print(f"Test participant {p_id} - Current: {p_row['Current_Region']} → Requested: {p_row['Requested_Region']} → Derived: {p_row['Derived_Region']}")
                    print(f"  Status: {p_row.get('Status', 'Unknown')}, Raw Status: {p_row.get('Raw_Status', 'Unknown')}")
    
    # Add score calculation fields
    normalized_df = calculate_preference_scores(normalized_df)
    
    return normalized_df

def standardize_time_preference(time_pref):
    """
    Standardize time preference format
    
    Args:
        time_pref: Time preference string
        
    Returns:
        Standardized time preference string with PLURAL time formats (Evenings, Days)
    """
    if pd.isna(time_pref) or time_pref == '':
        return ''
    
    time_pref = str(time_pref).strip()
    
    # Map common variations to standard format
    day_mapping = {
        'monday': 'Monday',
        'tuesday': 'Tuesday',
        'wednesday': 'Wednesday',
        'thursday': 'Thursday',
        'friday': 'Friday',
        'saturday': 'Saturday',
        'sunday': 'Sunday',
        'm-th': 'Monday-Thursday',
        'm-f': 'Monday-Friday',
        'mon': 'Monday',
        'tue': 'Tuesday',
        'wed': 'Wednesday',
        'thu': 'Thursday',
        'fri': 'Friday',
        'sat': 'Saturday',
        'sun': 'Sunday'
    }
    
    # Updated time mapping to keep Mornings, Afternoons, and Evenings distinct
    time_mapping = {
        'morning': 'Mornings',
        'mornings': 'Mornings', 
        'afternoon': 'Afternoons',
        'afternoons': 'Afternoons',
        'evening': 'Evenings',  # Singular to plural
        'evenings': 'Evenings',
        'day': 'Days',  # Singular to plural
        'days': 'Days',
        'daytime': 'Days',
        'night': 'Evenings',
        'daylight': 'Days'
    }
    
    # Check if the pattern is "Day (Time)"
    pattern = r'(.*?)\s*\((.*?)\)'
    match = re.search(pattern, time_pref)
    
    if match:
        day_part = match.group(1).strip().lower()
        time_part = match.group(2).strip().lower()
        
        standardized_day = day_mapping.get(day_part, day_part.capitalize())
        standardized_time = time_mapping.get(time_part, time_part.capitalize())
        
        # If time part wasn't found in mapping, check if it's a singular that needs pluralizing
        if standardized_time == time_part.capitalize():
            if standardized_time == 'Evening':
                standardized_time = 'Evenings'
            elif standardized_time == 'Day':
                standardized_time = 'Days'
        
        return f"{standardized_day} ({standardized_time})"
    
    # For patterns without parentheses, try to identify and standardize time references
    lower_pref = time_pref.lower()
    if 'evening' in lower_pref:
        return time_pref.replace('evening', 'Evenings').replace('Evening', 'Evenings')
    elif 'day' in lower_pref and 'day' != lower_pref:
        return time_pref.replace('day', 'Days').replace('Day', 'Days')
        
    # If no specific pattern is found, just capitalize and return
    return time_pref.capitalize()

def is_time_compatible(time1, time2, is_important=False, is_continuing_member=False, is_circle_time=False):
    """
    Check if two time preferences are compatible, including day ranges
    
    Args:
        time1: First time preference string
        time2: Second time preference string
        is_important: Boolean indicating whether to print detailed debug logs
        is_continuing_member: Boolean indicating if this is a continuing member's time preference
        is_circle_time: Boolean indicating if time2 is the official circle meeting time
        
    Returns:
        Boolean indicating if the time preferences are compatible
    """
    # Only debug specific cases
    important_pairs = [
        # Circle IP-SIN-01 with time "Varies (Evenings)" and participant 73177784103
        ("Varies (Evenings)", "Monday-Thursday (Evenings)"),
        ("Monday-Thursday (Evenings)", "Varies (Evenings)"),
        # Circle IP-LON-04 with time "Tuesday (Evenings)" and participant 50625303450
        ("Tuesday (Evenings)", "Monday-Thursday (Evenings)"),
        ("Monday-Thursday (Evenings)", "Tuesday (Evenings)"),
        # Circle IP-HOU-02 with time "M-Th (Evenings)" and Houston participants
        ("M-Th (Evenings)", "Monday-Thursday (Evenings)"),
        ("Monday-Thursday (Evenings)", "M-Th (Evenings)"),
        ("M-Th (Evenings)", "Monday-thursday (Evenings)"),
        ("Monday-thursday (Evenings)", "M-Th (Evenings)"),
        # Also check Monday-thursday variants (lowercase t)
        ("Varies (Evenings)", "Monday-thursday (Evenings)"),
        ("Monday-thursday (Evenings)", "Varies (Evenings)"),
        ("Tuesday (Evenings)", "Monday-thursday (Evenings)"),
        ("Monday-thursday (Evenings)", "Tuesday (Evenings)"),
        # Add Seattle-specific cases
        ("Wednesday (Evenings)", "Monday-Thursday (Evenings)"),
        ("Monday-Thursday (Evenings)", "Wednesday (Evenings)"),
        ("Wednesday (Evenings)", "Monday-thursday (Evenings)"),
        ("Monday-thursday (Evenings)", "Wednesday (Evenings)")
    ]
    
    # Check if this is a known problematic case to force debug
    if not is_important:
        # Check more flexibly to catch casing variations
        time1_lower = str(time1).lower() if not pd.isna(time1) else ""
        time2_lower = str(time2).lower() if not pd.isna(time2) else ""
        
        # Check for specific patterns we care about
        if ("varies" in time1_lower and "evening" in time1_lower and "monday" in time2_lower and "thursday" in time2_lower) or \
           ("varies" in time2_lower and "evening" in time2_lower and "monday" in time1_lower and "thursday" in time1_lower) or \
           ("tuesday" in time1_lower and "evening" in time1_lower and "monday" in time2_lower and "thursday" in time2_lower) or \
           ("tuesday" in time2_lower and "evening" in time2_lower and "monday" in time1_lower and "thursday" in time1_lower) or \
           ("wednesday" in time1_lower and "evening" in time1_lower and "monday" in time2_lower and "thursday" in time2_lower) or \
           ("wednesday" in time2_lower and "evening" in time2_lower and "monday" in time1_lower and "thursday" in time1_lower):
            is_important = True
        else:
            # Check exact matches from the list
            is_important = (time1, time2) in important_pairs
    
    if is_important:
        print(f"\n🔍 ENHANCED COMPATIBILITY CHECK between '{time1}' and '{time2}'")
        if is_continuing_member:
            print(f"  ℹ️ This is a continuing member's time preference")
        if is_circle_time:
            print(f"  ℹ️ The second time preference is the official circle meeting time")
    
    # ENHANCED FIX: SPECIAL CASE FOR CONTINUING CIRCLES
    # If this is a continuing member and time2 is the circle time,
    # ALL time values for the member should be considered compatible with their circle's time
    # This ensures CURRENT-CONTINUING members stay in their circles regardless of time preferences
    if is_continuing_member and is_circle_time:
        if is_important:
            print(f"  ✅ CRITICAL FIX: CONTINUING member automatically compatible with their circle time '{time2}'")
            if pd.isna(time1) or time1 == '':
                print(f"     (Member has empty/missing time preference)")
            else:
                print(f"     (Member has time preference '{time1}' but we're ignoring it)")
        return True
    
    # CRITICAL FIX: Special handling for Wednesday in Monday-Thursday ranges
    # This ensures compatibility between specific days and day ranges that include them
    if is_circle_time and not pd.isna(time1) and not pd.isna(time2):
        time1_lower = str(time1).lower()
        time2_lower = str(time2).lower()
        
        # Check if one preference is Wednesday and the other is Monday-Thursday
        if ('wednesday' in time1_lower and ('monday-thursday' in time2_lower or 'm-th' in time2_lower)) or \
           ('wednesday' in time2_lower and ('monday-thursday' in time1_lower or 'm-th' in time1_lower)):
            # Check if both have same time of day (morning, evening, etc.)
            if ('evening' in time1_lower and 'evening' in time2_lower) or \
               ('morning' in time1_lower and 'morning' in time2_lower) or \
               ('day' in time1_lower and 'day' in time2_lower):
                if is_important:
                    print(f"  ✅ COMPATIBLE - Wednesday is within Monday-Thursday range with matching time of day")
                return True
    
    # Handle None, NaN or empty strings (normal case)
    if pd.isna(time1) or pd.isna(time2) or time1 == '' or time2 == '':
        if is_important:
            print(f"  ❌ INCOMPATIBLE - One or both inputs is empty or invalid")
        return False
    
    # Standardize both time preferences
    std_time1 = standardize_time_preference(time1)
    std_time2 = standardize_time_preference(time2)
    
    if is_important:
        print(f"  Standardized to: '{std_time1}' and '{std_time2}'")
    
    # Direct string match after standardization?
    if std_time1 == std_time2:
        if is_important:
            print(f"  ✅ COMPATIBLE - Direct string match after standardization")
        return True
    
    # Special handling for "Varies" - compatible with anything
    # This makes the matching more flexible
    # Check for "Varies" in standard format and also any occurrence of "Varies" in either part
    if any(v in s.lower() for s in [std_time1.lower(), std_time2.lower()] for v in ["varies (varies)", "varies ("]):
        if is_important:
            print(f"  ✅ COMPATIBLE - One preference includes 'Varies' which matches anything")
        return True
        
    # Also check if "Varies" appears anywhere in the string (more flexible matching)
    if any("varies" in s.lower() for s in [time1, time2, std_time1, std_time2]):
        if is_important:
            print(f"  ✅ COMPATIBLE - One preference contains 'Varies' which improves flexibility")
        return True
        
    # Enhanced flexibility - if one has "Any" in it, it's compatible
    if any("any" in s.lower() for s in [time1, time2, std_time1, std_time2]):
        if is_important:
            print(f"  ✅ COMPATIBLE - One preference includes 'Any' which matches anything")
        return True
        
    # CRITICAL FIX: Explicit check for Wednesday vs Monday-Thursday compatibility
    # This is a common case that's causing problems with Seattle circles
    time1_lower = str(time1).lower()
    time2_lower = str(time2).lower()
    
    if ('wednesday' in time1_lower and 'evening' in time1_lower and 
        ('monday-thursday' in time2_lower or 'm-th' in time2_lower) and 'evening' in time2_lower) or \
       ('wednesday' in time2_lower and 'evening' in time2_lower and 
        ('monday-thursday' in time1_lower or 'm-th' in time1_lower) and 'evening' in time1_lower):
        if is_important:
            print(f"  ✅ COMPATIBLE - Direct match on Wednesday(Evenings) with Monday-Thursday(Evenings)")
        return True
    
    # Define helper functions for extracting time components
    def extract_time_components(time_str):
        """Extract day part and time period from a time preference string"""
        # Split into day part and time period
        parts = time_str.split('(')
        day_part = parts[0].strip()
        time_period = parts[1].replace(')', '').strip() if len(parts) > 1 else ''
        
        # Special debug for specific time preferences we're having issues with
        if is_important or (
           (day_part == 'Varies' and time_period == 'Evenings') or 
           ('Monday' in day_part and 'Thursday' in day_part and time_period == 'Evenings') or
           (day_part == 'Tuesday' and time_period == 'Evenings')):
            print(f"DEBUG: Extracted '{day_part}' and '{time_period}' from '{time_str}'")
        
        return day_part, time_period
    
    # Extract day parts and time periods
    day_part1, time_period1 = extract_time_components(std_time1)
    day_part2, time_period2 = extract_time_components(std_time2)
    
    if is_important:
        print(f"  Time periods: '{time_period1}' vs '{time_period2}'")
        print(f"  Day parts: '{day_part1}' vs '{day_part2}'")
    
    # Special case handling for "Varies" in day part
    if day_part1.lower() == 'varies' or day_part2.lower() == 'varies':
        # Varies matches any day part
        day_match = True
        if is_important:
            print(f"  ✓ Day parts match: One is 'Varies' which matches any day")
    else:
        # Normal day matching will be done below
        day_match = None
    
    # Check time period compatibility - now with distinct Mornings, Afternoons, Evenings, Days
    time_period_match = False
    if time_period1.lower() == 'varies' or time_period2.lower() == 'varies':
        time_period_match = True
        if is_important:
            print(f"  ✓ Time periods match: One is 'Varies' which matches any time period")
    else:
        # Exact match required - Mornings != Afternoons != Evenings != Days
        time_period_match = time_period1.lower() == time_period2.lower()
        if is_important:
            print(f"  {'✓' if time_period_match else '✗'} Time periods {'' if time_period_match else 'do not '}match")
            if not time_period_match:
                print(f"    Note: '{time_period1}' and '{time_period2}' are treated as distinct time periods")
    
    if not time_period_match:
        if is_important:
            print(f"  ❌ INCOMPATIBLE - Time periods don't match")
        return False
    
    # Skip day extraction if we already know days match
    if day_match is True:
        if is_important:
            print(f"  ✅ FINAL RESULT: COMPATIBLE - Time periods match and day part contains 'Varies'")
        return True
    
    # Extract days function with enhanced handling for all formats
    def extract_days(day_part):
        """Extract individual days from day part with support for ranges and abbreviations"""
        # Standard day name mapping
        all_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        all_days_lower = [d.lower() for d in all_days]
        
        # Special abbreviation mapping
        abbr_map = {
            'm': 'monday',
            'mon': 'monday',
            't': 'tuesday',
            'tue': 'tuesday',
            'tues': 'tuesday',
            'w': 'wednesday',
            'wed': 'wednesday',
            'th': 'thursday',
            'thur': 'thursday',
            'thurs': 'thursday',
            'f': 'friday',
            'fri': 'friday',
            's': 'saturday',
            'sa': 'saturday',
            'sat': 'saturday',
            'su': 'sunday',
            'sun': 'sunday'
        }
        
        # Special case: Varies matches all days
        if day_part.lower() == 'varies':
            if is_important:
                print(f"    Day part 'Varies' expands to all days of the week")
            return all_days
            
        # Special case: Weekend expansion
        if day_part.lower() == 'weekend':
            if is_important:
                print(f"    Day part 'Weekend' expands to Saturday and Sunday")
            return ['Saturday', 'Sunday']
            
        # Special case: Monday-Friday expansion
        if day_part.lower() in ['monday-friday', 'mon-fri', 'm-f', 'm-friday', 'monday-fri']:
            if is_important:
                print(f"    Day part '{day_part}' expands to Monday through Friday")
            return ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            
        # Special case: M-Th format (handle case variations)
        if day_part.lower() in ['m-th', 'm-thu', 'm-thur', 'm-thurs', 'm-thursday']:
            if is_important:
                print(f"    Day part '{day_part}' expands to Monday through Thursday")
            return ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
        
        # Special case: handle "monday-thursday" explicitly (including lowercase t issue)
        if day_part.lower() in ['monday-thursday', 'monday-thurs', 'mon-thurs', 'mon-thu', 'mon-thursday', 'monday-thu', 'mon-t', 'monday-t']:
            if is_important:
                print(f"    Day part '{day_part}' expands to Monday through Thursday")
            return ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
            
        # Special case for specific days that we know should be compatible with our test cases
        if day_part.lower() == 'tuesday' and any(range_day in [d.lower() for d in [day_part1, day_part2]] for range_day in ['monday-thursday', 'monday-thurs', 'monday-thursday', 'monday-thu']):
            if is_important:
                print(f"    Special case: Tuesday is within Monday-Thursday range")
            return ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
            
        # Special case for Wednesday compatibility with Monday-Thursday range
        if day_part.lower() == 'wednesday' and any(range_day in [d.lower() for d in [day_part1, day_part2]] for range_day in ['monday-thursday', 'monday-thurs', 'monday-thursday', 'monday-thu']):
            if is_important:
                print(f"    Special case: Wednesday is within Monday-Thursday range")
            return ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
            
        # The reverse case: Monday-Thursday containing Wednesday
        if 'monday-thursday' in day_part.lower() and any(day in [d.lower() for d in [day_part1, day_part2]] for day in ['wednesday']):
            if is_important:
                print(f"    Special case: Monday-Thursday range contains Wednesday")
            return ['Monday', 'Tuesday', 'Wednesday', 'Thursday']
            
        # Handle day ranges with dash notation
        days = []
        if '-' in day_part:
            # Extract start and end days
            day_range = day_part.split('-')
            start_day = day_range[0].strip().lower()
            end_day = day_range[1].strip().lower()
            
            # Try to map abbreviations to full day names
            if start_day in abbr_map:
                start_day = abbr_map[start_day]
                if is_important:
                    print(f"    Mapped abbreviation {day_range[0].strip()} to {start_day}")
                
            if end_day in abbr_map:
                end_day = abbr_map[end_day]
                if is_important:
                    print(f"    Mapped abbreviation {day_range[1].strip()} to {end_day}")
            
            # Get indices for range lookup
            try:
                start_idx = all_days_lower.index(start_day)
                end_idx = all_days_lower.index(end_day)
                
                # Extract the range with proper capitalization
                days = all_days[start_idx:end_idx+1]
                if is_important:
                    print(f"    Day range {day_range[0].strip()}-{day_range[1].strip()} expands to {days}")
            except ValueError:
                if is_important:
                    print(f"    Could not parse day range {day_part}, using as-is")
                # Special case for 'thursday' if ValueError was raised trying to find it
                if 'thursday' in end_day:
                    end_day = 'thursday'  # Make sure it's lowercase for matching
                    try:
                        start_idx = all_days_lower.index(start_day)
                        end_idx = all_days_lower.index(end_day)
                        days = all_days[start_idx:end_idx+1]
                        if is_important:
                            print(f"    Second attempt: Day range {day_range[0].strip()}-{day_range[1].strip()} expands to {days}")
                    except ValueError:
                        days = [day_part]
                else:
                    days = [day_part]
        else:
            # Single day - handle abbreviations and capitalization
            day_lower = day_part.lower()
            
            # Check for abbreviation
            if day_lower in abbr_map:
                day_lower = abbr_map[day_lower]
                if is_important:
                    print(f"    Mapped abbreviation {day_part} to {day_lower}")
            
            # Look up proper capitalization
            try:
                idx = all_days_lower.index(day_lower)
                days = [all_days[idx]]
                if is_important:
                    print(f"    Normalized day {day_part} to {days[0]}")
            except ValueError:
                days = [day_part]
                if is_important:
                    print(f"    Could not normalize day {day_part}, using as-is")
                
        return days
    
    # Extract individual days for both preferences
    days1 = extract_days(day_part1)
    days2 = extract_days(day_part2)
    
    if is_important:
        print(f"  Expanded days: {days1} vs {days2}")
    
    # Check for day overlap
    overlap = False
    common_days = set(days1).intersection(set(days2))
    if common_days:
        overlap = True
        if is_important:
            print(f"  ✓ Days overlap: Common days are {sorted(common_days)}")
    else:
        if is_important:
            print(f"  ✗ No overlapping days found")
        
    result = time_period_match and overlap
    if is_important:
        print(f"  {'✅' if result else '❌'} FINAL RESULT: {'' if result else 'IN'}COMPATIBLE - Time periods {'do not ' if not time_period_match else ''}match and days {'do not ' if not overlap else ''}overlap")
    
    return result

def calculate_preference_scores(df):
    """
    Calculate preference scores for location and time matches
    
    Args:
        df: DataFrame with participant data
        
    Returns:
        DataFrame with added preference score columns
    """
    result_df = df.copy()
    
    # Initialize score columns if they don't exist
    # Note: These scores will be recalculated later in optimizer.py
    # based on actual circle assignments
    if 'location_score' not in result_df.columns:
        result_df['location_score'] = 0
    
    if 'time_score' not in result_df.columns:
        result_df['time_score'] = 0
    
    if 'total_score' not in result_df.columns:
        result_df['total_score'] = 0
        
    # Note: We're only initializing the score columns here
    # The actual scores will be calculated after circle assignments in optimizer.py
    # This ensures the scores reflect how well the assigned circle matches preferences
    
    return result_df
