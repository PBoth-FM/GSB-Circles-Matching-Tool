import pandas as pd
import numpy as np
from utils.normalization import normalize_regions, normalize_subregions
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_co_leader_max_new_members(value):
    """
    Process co-leader max new members value
    
    Args:
        value: The co-leader max new members value to process
        
    Returns:
        Processed value (None, string 'None', or integer)
    """
    # Handle None/NaN values
    if pd.isna(value) or value is None or value == '':
        return None
    
    # Handle 'None' literal string
    if isinstance(value, str) and value.strip().lower() == 'none':
        return 'None'
    
    # Try to convert to integer
    try:
        int_value = int(value)
        # Handle 0 as equivalent to 'None' - no new members should be added
        if int_value == 0:
            return 'None'
        return int_value
    except (ValueError, TypeError):
        # If not a valid number, return None
        return None

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

def process_data(df):
    """
    Process and clean the participant data
    
    Args:
        df: Pandas DataFrame with participant data
        
    Returns:
        Processed DataFrame
    """
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
        processed_df['Encoded ID'] = processed_df['Encoded ID'].astype(str)
    
    # Process co-leader max new members if the column exists
    if 'co_leader_max_new_members' in processed_df.columns:
        processed_df['co_leader_max_new_members'] = processed_df['co_leader_max_new_members'].apply(
            process_co_leader_max_new_members
        )
    else:
        # Add the column with default None values if it doesn't exist
        processed_df['co_leader_max_new_members'] = None
    
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

def normalize_data(df):
    """
    Normalize region and subregion data
    
    Args:
        df: Pandas DataFrame with processed participant data
        
    Returns:
        DataFrame with normalized region and subregion data
    """
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
    
    # Normalize subregions
    for location_col in ['first_choice_location', 'second_choice_location', 'third_choice_location', 'Current_Subregion']:
        if location_col in normalized_df.columns:
            normalized_df[location_col] = normalized_df[location_col].apply(
                lambda x: normalize_subregions(x) if pd.notna(x) and x != '' else x
            )
    
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
        normalized_df['Derived_Region'] = normalized_df.apply(
            lambda row: row['Current_Region'] if row['Status'] == 'CURRENT-CONTINUING' else row['Requested_Region'], 
            axis=1
        )
    
    # Add score calculation fields
    normalized_df = calculate_preference_scores(normalized_df)
    
    return normalized_df

def standardize_time_preference(time_pref):
    """
    Standardize time preference format
    
    Args:
        time_pref: Time preference string
        
    Returns:
        Standardized time preference string
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
    
    time_mapping = {
        'morning': 'Morning',
        'afternoon': 'Afternoon',
        'evening': 'Evening',
        'evenings': 'Evening',
        'day': 'Day',
        'days': 'Day'
    }
    
    # Check if the pattern is "Day (Time)"
    pattern = r'(.*?)\s*\((.*?)\)'
    match = re.search(pattern, time_pref)
    
    if match:
        day_part = match.group(1).strip().lower()
        time_part = match.group(2).strip().lower()
        
        standardized_day = day_mapping.get(day_part, day_part.capitalize())
        standardized_time = time_mapping.get(time_part, time_part.capitalize())
        
        return f"{standardized_day} ({standardized_time})"
    
    # If no pattern, just return as is with first letter capitalized
    return time_pref.capitalize()

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
