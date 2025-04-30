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

        # SUPER DETAILED DIAGNOSTICS
        print("\n🔬🔬🔬 SUPER DETAILED DATA ANALYSIS IN LOAD_DATA 🔬🔬🔬")
        print(f"🔬 Raw DataFrame shape: {df.shape}")
        print(f"🔬 Raw columns before mapping: {df.columns.tolist()}")

        # Check for circle-related columns
        circle_id_columns = [col for col in df.columns if "circle" in col.lower() and "id" in col.lower()]
        if circle_id_columns:
            print(f"🔬 Potential circle ID columns in raw data: {circle_id_columns}")

            # Check for sample values
            for col in circle_id_columns:
                non_null = df[df[col].notna()]
                if len(non_null) > 0:
                    sample_values = non_null[col].unique()[:5]
                    print(f"🔬 Sample values for '{col}': {list(sample_values)}")

        # Check for status columns
        status_columns = [col for col in df.columns if "status" in col.lower()]
        if status_columns:
            print(f"🔬 Potential status columns in raw data: {status_columns}")
            for col in status_columns:
                if col in df.columns:
                    status_counts = df[col].value_counts().to_dict()
                    print(f"🔬 '{col}' value counts: {status_counts}")

        # Check for test circles in any column
        test_circles = ['IP-SIN-01', 'IP-LON-04', 'IP-HOU-02']
        for test_circle in test_circles:
            for col in df.columns:
                if df[col].astype(str).str.contains(test_circle).any():
                    matches = df[df[col].astype(str).str.contains(test_circle)]
                    print(f"🔬 TEST CIRCLE {test_circle} found in column '{col}' ({len(matches)} matches)")

        print("🔬🔬🔬 DETAILED DIAGNOSTICS BEFORE MAPPING 🔬🔬🔬\n")

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
        'Current Circle ID': 'Current_Circle_ID',  # Changed to preserve capitalization pattern
        'Current Subregion': 'Current_Subregion',
        'Current Meeting Time': 'Current_Meeting_Time',
        # Current Co-Leader status
        'Current Co-Leader': 'Current_Co_Leader',
        'Co-Leader Status': 'Current_Co_Leader',
        'Current Circle Co-Leader': 'Current_Co_Leader',
        # New column for max new members
        'Co-Leader Max New Members': 'co_leader_max_new_members',
        'Co-Leader Maximum New Members': 'co_leader_max_new_members',
        'Maximum New Members Requested': 'co_leader_max_new_members',
        'Max New Members': 'co_leader_max_new_members',
        'Co-Leader Response: Max New Member Requested': 'co_leader_max_new_members'
    }

    # Debug information about column mapping
    print(f"Input column names: {df.columns.tolist()}")
    print(f"Column mapping: {column_mapping}")

    # Check for any columns in the source data that might match our expected "Current Circle ID"
    for col in df.columns:
        if "circle" in col.lower() and "id" in col.lower():
            print(f"Found potential circle ID column: '{col}'")
            # Count non-null values in this column
            non_null_count = df[col].notna().sum()
            print(f"  Number of non-null values: {non_null_count}")
            if 'Status' in df.columns:
                continuing_count = df[df['Status'] == 'CURRENT-CONTINUING'][col].notna().sum()
                print(f"  Number of CURRENT-CONTINUING with non-null value: {continuing_count}")

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

    # Define detailed status mapping for Raw_Status column
    detailed_status_mapping = {
        # Current variations
        'CURRENT-CONTINUING': 'Current-CONTINUING',
        'Current-CONTINUING': 'Current-CONTINUING',
        'Current-CONTINUING ': 'Current-CONTINUING',

        # Moving within region
        'Current-MOVING within Region': 'Current-MOVING within Region',

        # Moving out variations
        'Current-MOVING OUT of Region': 'Current-MOVING OUT of Region',
        'Current - MOVING OUT OF region': 'Current-MOVING OUT of Region',

        # Moving into variations
        'Current-MOVING INTO Region': 'Current-MOVING INTO Region',
        'Current - MOVING INTO region': 'Current-MOVING INTO Region',
        'Current-MOVING INTO Region ': 'Current-MOVING INTO Region',

        # New participants variations
        'NEW to Circles': 'NEW to Circles',
        'Requesting 2nd Circle': 'Requesting 2nd Circle',
        'xNEW to Circles (Waitlist)': 'NEW to Circles (Waitlist)',
        'NEW to Circles (Waitlist) Requesting 2nd Circle': 'NEW to Circles (Waitlist) Requesting 2nd Circle'
    }

    # Binary status mapping for algorithm compatibility
    binary_status_mapping = {
        # Continuing in their circle
        'CURRENT-CONTINUING': 'CURRENT-CONTINUING',
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

    # Create Raw_Status column with detailed status mapping
    normalized_df['Raw_Status'] = normalized_df['Status'].apply(
        lambda x: detailed_status_mapping.get(x.strip() if isinstance(x, str) else x, x)
    )

    # Apply binary mapping to maintain compatibility with existing code
    normalized_df['Status'] = normalized_df['Status'].apply(
        lambda x: binary_status_mapping.get(x.strip() if isinstance(x, str) else x, x)
    )

    # Filter out NOT Continuing and MOVING OUT participants
    not_continuing_mask = normalized_df['Status'].str.contains('NOT Continuing', case=False, na=False)
    moving_out_mask = normalized_df['Status'].str.contains('MOVING OUT', case=False, na=False)

    # Count records to be filtered
    not_continuing_count = sum(not_continuing_mask)
    moving_out_count = sum(moving_out_mask)

    # Apply filters
    normalized_df = normalized_df[~(not_continuing_mask | moving_out_mask)]

    # Store counts in session state for display
    import streamlit as st
    if 'status_filter_counts' not in st.session_state:
        st.session_state.status_filter_counts = {}
    st.session_state.status_filter_counts['not_continuing'] = not_continuing_count
    st.session_state.status_filter_counts['moving_out'] = moving_out_count

    # Handle region for 'Current-MOVING within Region'
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

    # Validate Status values - adjusted to include both binary and detailed statuses
    if 'Status' in df.columns:
        valid_binary_statuses = ['CURRENT-CONTINUING', 'NEW', 'MOVING OUT', 'WAITLIST']
        invalid_statuses = df[~df['Status'].isin(valid_binary_statuses)]['Status'].unique()
        if len(invalid_statuses) > 0:
            errors.append(f"Invalid binary Status values: {', '.join(map(str, invalid_statuses))}")

    # No validation needed for Raw_Status since it contains the detailed values

    # Validate host volunteer information
    if 'host' in df.columns:
        valid_host_values = ['Always', 'Sometimes', 'Never Host', 'n/a', '']
        invalid_host_values = df[~df['host'].isin(valid_host_values)]['host'].unique()
        if len(invalid_host_values) > 0:
            errors.append(f"Invalid host values: {', '.join(map(str, invalid_host_values))}")

    return errors