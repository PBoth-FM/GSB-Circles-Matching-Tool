import pandas as pd
import numpy as np
import io
import time
import re

def format_time_elapsed(seconds):
    """
    Format elapsed time in a human-readable format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)} min {int(seconds)} sec"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)} hr {int(minutes)} min {int(seconds)} sec"

def generate_download_link(df):
    """
    Generate a downloadable link for a DataFrame with properly ordered columns

    Args:
        df: Pandas DataFrame to convert

    Returns:
        CSV data as string with reordered columns
    """
    # Create a copy to avoid modifying the original DataFrame
    output_df = df.copy()

    # Sort by Derived_Region and proposed_NEW_circles_id, with nulls at end
    if 'Derived_Region' in output_df.columns and 'proposed_NEW_circles_id' in output_df.columns:
        output_df = output_df.sort_values(
            ['Derived_Region', 'proposed_NEW_circles_id'],
            na_position='last'
        )

    # ENHANCED DIAGNOSTICS: Count participants in CSV before processing
    print("\n🔍 🔍 🔍 CSV GENERATION DIAGNOSTICS - ENHANCED DEBUG 🔍 🔍 🔍")
    print(f"  Initial DataFrame shape: {output_df.shape}")

    # Debug: Check for specific columns we expect to have issues with
    meta_columns = ['proposed_NEW_Subregion', 'proposed_NEW_DayTime', 'proposed_NEW_circles_id']
    for col in meta_columns:
        if col in output_df.columns:
            print(f"  ✓ Column '{col}' exists in dataframe")
        else:
            print(f"  ⚠️ Column '{col}' NOT FOUND in dataframe!")

    # Count Unknown values in key columns before any fixes
    if 'proposed_NEW_Subregion' in output_df.columns:
        unknown_count = output_df[output_df['proposed_NEW_Subregion'] == 'Unknown'].shape[0]
        print(f"  PRE-FIX: Found {unknown_count} rows with 'Unknown' subregion value")

    if 'proposed_NEW_DayTime' in output_df.columns:
        unknown_count = output_df[output_df['proposed_NEW_DayTime'] == 'Unknown'].shape[0]
        print(f"  PRE-FIX: Found {unknown_count} rows with 'Unknown' meeting time value")

    # Check for matched vs unmatched before any filtering
    if 'proposed_NEW_circles_id' in output_df.columns:
        valid_circle_mask = (output_df['proposed_NEW_circles_id'].notna()) & (output_df['proposed_NEW_circles_id'] != 'UNMATCHED')
        matched_count = len(output_df[valid_circle_mask])
        unmatched_count = len(output_df[output_df['proposed_NEW_circles_id'] == 'UNMATCHED'])
        print(f"  CSV Pre-processing - Matched: {matched_count}, Unmatched: {unmatched_count}")

        # Detailed counts by circle ID
        circle_counts = output_df['proposed_NEW_circles_id'].value_counts().to_dict()
        print(f"  Top circles (first 5): {dict(list(circle_counts.items())[:5])}")

        # Look for test circle IDs in the assigned circles
        test_circles = [c_id for c_id in output_df['proposed_NEW_circles_id'].unique() 
                        if isinstance(c_id, str) and any(pattern in c_id for pattern in ['IP-TEST', 'IP-SIN-01', 'IP-LON-04', 'IP-HOU-02'])]
        if test_circles:
            print(f"  ⚠️ CSV contains {len(test_circles)} TEST CIRCLE IDs: {test_circles}")
            # Count participants in test circles
            test_participants = len(output_df[output_df['proposed_NEW_circles_id'].isin(test_circles)])
            print(f"  ⚠️ {test_participants} participants assigned to test circles in CSV")

            # Calculate adjusted totals without test circles
            adjusted_matched = matched_count - test_participants
            print(f"  Adjusted matched count (removing test participants): {adjusted_matched}")

        # CRITICAL FIX: Apply the same metadata fixes to the CSV output
        # Check if we have unknown values in subregion or meeting time columns
        if 'proposed_NEW_Subregion' in output_df.columns or 'proposed_NEW_DayTime' in output_df.columns:

            # ENHANCED FIX: Normalize all subregion values using standardized normalization tables
            try:
                # Import normalize_subregion function from circle_reconstruction
                from modules.circle_reconstruction import normalize_subregion, clear_normalization_cache

                # Clear the normalization cache to ensure fresh data
                clear_normalization_cache()
                print("\n🔄 NORMALIZING SUBREGION VALUES IN CSV EXPORT")

                # Track special problem regions for enhanced logging
                problem_subregions = ['Napa Valley', 'North Spokane', 'Unknown']
                normalized_count = 0
                problem_fixed = 0

                # Apply normalization to proposed_NEW_Subregion column
                if 'proposed_NEW_Subregion' in output_df.columns:
                    # Create a separate Series for comparison
                    original_values = output_df['proposed_NEW_Subregion'].copy()

                    # Apply normalization
                    output_df['proposed_NEW_Subregion'] = output_df['proposed_NEW_Subregion'].apply(
                        lambda x: normalize_subregion(x) if pd.notnull(x) else x
                    )

                    # Count changes
                    changed_mask = original_values != output_df['proposed_NEW_Subregion']
                    normalized_count = changed_mask.sum()

                    # Count problem region fixes
                    if normalized_count > 0:
                        for problem in problem_subregions:
                            problem_mask = (original_values == problem) & changed_mask
                            problem_count = problem_mask.sum()
                            if problem_count > 0:
                                problem_fixed += problem_count
                                print(f"  ✅ Fixed {problem_count} instances of '{problem}' in CSV export")

                    print(f"  Normalized {normalized_count} subregion values in CSV export")

            except Exception as e:
                print(f"  ⚠️ Error normalizing subregion values in CSV: {str(e)}")

            # Check for unknown values
            unknown_subregions = 0
            unknown_meeting_times = 0

            if 'proposed_NEW_Subregion' in output_df.columns:
                unknown_subregions = output_df[output_df['proposed_NEW_Subregion'] == 'Unknown'].shape[0]

            if 'proposed_NEW_DayTime' in output_df.columns:
                unknown_meeting_times = output_df[output_df['proposed_NEW_DayTime'] == 'Unknown'].shape[0]

            print(f"\n  🔍 CSV METADATA CHECK: Found {unknown_subregions} Unknown subregions and {unknown_meeting_times} Unknown meeting times")

            if unknown_subregions > 0 or unknown_meeting_times > 0:
                print("  🔧 APPLYING CENTRALIZED METADATA FIXES TO CSV OUTPUT")

                # Use our centralized metadata fix function
                try:
                    from utils.metadata_manager import fix_participant_metadata_in_results
                    # Apply the centralized fix function
                    fixed_df = fix_participant_metadata_in_results(output_df)
                    # Update with fixed data
                    output_df = fixed_df
                    print("  ✅ Successfully applied centralized metadata fixes")
                except Exception as e:
                    print(f"  ⚠️ Error applying centralized metadata fixes: {str(e)}")
                    print("  Continuing with original metadata")

                # Final check for any remaining unknown values
                if 'proposed_NEW_Subregion' in output_df.columns:
                    remaining_unknown = output_df[output_df['proposed_NEW_Subregion'] == 'Unknown'].shape[0]
                    print(f"  Remaining unknown subregions: {remaining_unknown}")

                if 'proposed_NEW_DayTime' in output_df.columns:
                    remaining_unknown = output_df[output_df['proposed_NEW_DayTime'] == 'Unknown'].shape[0]
                    print(f"  Remaining unknown meeting times: {remaining_unknown}")
            else:
                print("  ✅ No Unknown metadata values found in CSV - no fixes needed")


    # Only keep columns that don't start with "Unnamed:"
    filtered_columns = [col for col in output_df.columns if not col.startswith('Unnamed:')]
    output_df = output_df[filtered_columns]
    print(f"  After filtering unnamed columns: {output_df.shape}")

    # CRITICAL FIX: Remove blank rows (rows with no Encoded ID)
    if 'Encoded ID' in output_df.columns:
        # Before filtering, let's identify any rows that have null Encoded ID but valid circle assignments
        if 'proposed_NEW_circles_id' in output_df.columns:
            valid_circle_mask_pre = (output_df['proposed_NEW_circles_id'].notna()) & (output_df['proposed_NEW_circles_id'] != 'UNMATCHED')
            null_id_but_matched = output_df[valid_circle_mask_pre & output_df['Encoded ID'].isna()]

            if len(null_id_but_matched) > 0:
                print(f"  ⚠️ FOUND THE MISSING PARTICIPANT(S): {len(null_id_but_matched)} rows have valid circle assignments but null Encoded ID")
                print(f"  These will be removed from the CSV but are counted in UI statistics:")
                for _, row in null_id_but_matched.iterrows():
                    circle_id = row['proposed_NEW_circles_id']
                    status = row.get('Status', 'Unknown')
                    raw_status = row.get('Raw_Status', 'Unknown')
                    print(f"  - Circle: {circle_id}, Status: {status}, Raw Status: {raw_status}")

                    # Show more details about this row to help identify it
                    if 'Last (Family) Name' in row and 'First (Given) Name' in row:
                        name = f"{row['Last (Family) Name']} {row['First (Given) Name']}"
                        print(f"    Name (if available): {name}")

                    # Print all non-null values for this row to help identify it
                    non_null_values = {col: val for col, val in row.items() if pd.notna(val) and not col.startswith('Unnamed:')}
                    print(f"    Key attributes: {list(non_null_values.keys())[:10]}")

        # Count blank rows before filtering
        blank_count = output_df['Encoded ID'].isna().sum()
        if blank_count > 0:
            print(f"  🔍 CRITICAL FIX: Found {blank_count} blank rows in results")
            # Keep only rows with a non-null Encoded ID
            output_df = output_df.dropna(subset=['Encoded ID'])
            print(f"  ✅ Removed {blank_count} blank rows from results CSV")

        # After removing blank rows, recount matched/unmatched
        if 'proposed_NEW_circles_id' in output_df.columns:
            valid_circle_mask = (output_df['proposed_NEW_circles_id'].notna()) & (output_df['proposed_NEW_circles_id'] != 'UNMATCHED')
            matched_count = len(output_df[valid_circle_mask])
            unmatched_count = len(output_df[output_df['proposed_NEW_circles_id'] == 'UNMATCHED'])
            print(f"  CSV Post-processing - Matched: {matched_count}, Unmatched: {unmatched_count}")

            # Check if we have stored UI IDs for comparison
            import streamlit as st
            if 'ui_matched_ids' in st.session_state:
                ui_ids = set(st.session_state.ui_matched_ids)
                csv_ids = set(output_df[valid_circle_mask]['Encoded ID'].tolist())

                # Find IDs in UI that are not in CSV
                ui_only_ids = ui_ids - csv_ids
                if ui_only_ids:
                    print(f"  ⚠️ Found {len(ui_only_ids)} IDs in UI statistics that are missing from CSV: {ui_only_ids}")

                # Find IDs in CSV that are not in UI (shouldn't happen, but check anyway)
                csv_only_ids = csv_ids - ui_ids
                if csv_only_ids:
                    print(f"  ⚠️ Found {len(csv_only_ids)} IDs in CSV that are not counted in UI statistics: {csv_only_ids}")

                print(f"  UI matched count: {len(ui_ids)}, CSV matched count: {len(csv_ids)}")

    # STEP 1: Remove specified columns by name
    columns_to_remove = ['region', 'participant_id']
    print(f"\n🔧 CSV COLUMN PROCESSING:")
    print(f"  Removing specified columns: {columns_to_remove}")
    
    for col_to_remove in columns_to_remove:
        if col_to_remove in output_df.columns:
            output_df = output_df.drop(columns=[col_to_remove])
            print(f"  ✅ Removed column: '{col_to_remove}'")
        else:
            print(f"  ⚠️ Column not found for removal: '{col_to_remove}'")

    # STEP 2: Define the exact column order as specified
    desired_column_order = [
        'Status',
        'Raw_Status',
        'Encoded ID',
        'proposed_NEW_circles_id',
        'unmatched_reason',
        'proposed_NEW_Subregion',
        'proposed_NEW_DayTime',
        'proposed_NEW_Coleader',
        'host_status_standardized',
        'host',
        'Current Co-Leader?',
        '(Non CLs) Volunteering to Co-Lead?',
        'Co-Leader Response:  CL in 2025?',
        'co_leader_max_new_members',
        'Derived_Region',
        'Region_Code',
        'Requested_Region',
        'Region Edit Made by tar',
        'If already in 2 Circles, list both regions',
        'If asking to join 2nd Circle, what is the 2nd region?',
        'If Can\'t Place In-Person, Open to Virtual-Only?',
        'first_choice_location',
        'second_choice_location',
        'third_choice_location',
        'location_score',
        'first_choice_time',
        'second_choice_time',
        'third_choice_time',
        'time_score',
        'total_score',
        'GSB Degree',
        'GSB_Class_Numeric',
        'GSB Class Year',
        'Class_Vintage',
        'Employment Status',
        'Industry Sector',
        'Racial Identity',
        'Relationship Status',
        'Children',
        'Sexual Orientation',
        'Gender Identity',
        'Special Needs?',
        'Current_Circle_ID',
        'Current_Region',
        'Current_Subregion',
        'Current/ Continuing Meeting Day',
        'Current/ Continuing Meeting Time',
        'Days & Times Additional Info',
        'Additional Comments From Form',
        'Co-Leader Response: Anything You are Looking for in New Members',
        'Co-Leader Response: Day(s)',
        'Co-Leader Response: Hosts',
        'Co-Leader Response: Meeting Format',
        'Co-Leader Response: Times',
        'Last (Family) Name',
        'First (Given) Name',
        'Preferred Email',
        'Mobile Phone',
        'Home City',
        'Home Country',
        'Home Phone',
        'Home State',
        'Business City',
        'Business Country',
        'Business Phone',
        'Business State',
        'Completed Form? Y/P'
    ]

    # STEP 3: Check which columns exist and log missing ones
    print(f"  Checking for expected columns:")
    ordered_columns = []
    missing_columns = []
    
    for col in desired_column_order:
        if col in output_df.columns:
            ordered_columns.append(col)
            print(f"  ✅ Found: '{col}'")
        else:
            missing_columns.append(col)
            print(f"  ⚠️ Missing: '{col}'")
    
    if missing_columns:
        print(f"  Total missing columns: {len(missing_columns)}")
    else:
        print(f"  ✅ All expected columns found!")

    # STEP 4: Add any remaining columns that weren't in the desired order
    remaining_columns = [col for col in output_df.columns 
                        if col not in ordered_columns 
                        and not col.startswith('Unnamed:')]
    
    if remaining_columns:
        print(f"  Additional columns not in specified order: {remaining_columns}")
        # Add them at the end, sorted alphabetically
        ordered_columns.extend(sorted(remaining_columns))

    # Create a new DataFrame with only the columns that exist
    final_columns = [col for col in ordered_columns if col in output_df.columns]
    final_df = output_df[final_columns]

    # CRITICAL FIX: Apply post-processing to fix any invalid circle IDs before CSV generation
    from utils.circle_id_postprocessor import has_unknown_circles, fix_unknown_circle_ids
    import streamlit as st

    if has_unknown_circles(final_df):
        print("\n🔧 PRE-CSV PROCESSING: Applying circle ID corrections before CSV generation")
        final_df = fix_unknown_circle_ids(final_df)

        # CRITICAL: Update session state with corrected data so Circle Composition table matches
        if hasattr(st.session_state, 'results') and st.session_state.results is not None:
            print("🔧 UPDATING SESSION STATE: Applying same corrections to session state data")
            st.session_state.results = fix_unknown_circle_ids(st.session_state.results)
            print("✅ Session state updated - Circle Composition table will now show corrected data")

    # Convert to CSV and return
    csv_buffer = io.StringIO()
    final_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def generate_circle_id(region, subregion, index, is_new=True):
    """
    Generate a circle ID following the naming convention

    Args:
        region: Region name
        subregion: Subregion name
        index: Circle index
        is_new: Whether this is a new circle (True) or existing circle (False)

    Returns:
        Circle ID string in format:
        - Virtual New circles: VO-{RegionCode}-NEW-{index} where RegionCode includes timezone (e.g., AM-GMT-6)
        - Virtual Existing circles: VO-{RegionCode}-{index}
        - In-person New circles: IP-{RegionCode}-NEW-{index}
        - In-person Existing circles: IP-{RegionCode}-{index}
    """
    # Import here to avoid circular imports
    from utils.normalization import get_region_code, get_region_code_with_subregion

    # Check if this is a virtual circle
    is_virtual = "Virtual" in str(region) if region is not None else False

    # Format the index as 2-digit number
    index_str = str(index).zfill(2)

    # Set the format prefix
    format_prefix = "VO" if is_virtual else "IP"

    # Get the appropriate region code
    if is_virtual and subregion:
        # For virtual circles, get the region code that includes timezone from subregion
        region_code = get_region_code_with_subregion(region, subregion, is_virtual=True)
        print(f"🔍 Virtual circle detected: region={region}, subregion={subregion}")
        print(f"🔍 Using region code with subregion: {region_code}")

        # CRITICAL FIX: Never allow UNKNOWN or Invalid region codes for virtual circles
        if region_code in ['UNKNOWN', 'Invalid', 'Unknown']:
            print(f"⚠️ CRITICAL FIX: Invalid region code '{region_code}' for virtual circle, applying fallback")
            if 'APAC+EMEA' in str(region):
                region_code = 'AE-GMT'
            elif 'Americas' in str(region):
                region_code = 'AM-GMT-5'
            else:
                region_code = 'AE-GMT'
            print(f"✅ Applied fallback region code: {region_code}")
    else:
        # For in-person circles, use the standard region code
        region_code = get_region_code(region)

        # CRITICAL FIX: Never allow UNKNOWN or Invalid region codes for any circles
        if region_code in ['UNKNOWN', 'Invalid', 'Unknown']:
            print(f"⚠️ CRITICAL FIX: Invalid region code '{region_code}', using fallback")
            region_code = 'NYC'  # Safe fallback for in-person circles

    # Format: {Format}-{RegionCode}-NEW-{index} for new circles
    # For existing circles, the format is {Format}-{RegionCode}-{index}
    if is_new:
        circle_id = f"{format_prefix}-{region_code}-NEW-{index_str}"
    else:
        circle_id = f"{format_prefix}-{region_code}-{index_str}"

    print(f"🔍 Generated circle ID: {circle_id} (is_virtual={is_virtual}, is_new={is_new})")
    return circle_id

def estimate_compatibility(participant, subregion, time_slot):
    """
    Estimate compatibility score between a participant and a potential circle

    Args:
        participant: Participant data (dict or Series)
        subregion: Circle subregion
        time_slot: Circle time slot

    Returns:
        Compatibility score (0-6)
    """
    score = 0

    # Location score (0-3)
    if participant.get('first_choice_location') == subregion:
        score += 3
    elif participant.get('second_choice_location') == subregion:
        score += 2
    elif participant.get('third_choice_location') == subregion:
        score += 1

    # Time score (0-3)
    if participant.get('first_choice_time') == time_slot:
        score += 3
    elif participant.get('second_choice_time') == time_slot:
        score += 2
    elif participant.get('third_choice_time') == time_slot:
        score += 1

    return score

def get_valid_participants(participants_df):
    """
    Filter DataFrame to only include valid participants with non-null Encoded IDs.

    Args:
        participants_df (DataFrame): DataFrame containing participant information

    Returns:
        DataFrame: Filtered DataFrame with only valid participants
    """
    # Handle both column name formats (for flexibility)
    id_col = 'Encoded ID' if 'Encoded ID' in participants_df.columns else 'Encoded_ID'

    # Enhanced filtering to exclude nan, None, empty strings, and 'nan' strings
    valid_mask = (
        participants_df[id_col].notna() & 
        (participants_df[id_col].astype(str) != 'None') &
        (participants_df[id_col].astype(str) != '') &
        (participants_df[id_col].astype(str).str.lower() != 'nan') &
        (participants_df[id_col].astype(str) != 'NaN') &
        (participants_df[id_col].astype(str).str.strip() != '')
    )
    
    valid_df = participants_df[valid_mask]

    # Log the filtering process for debugging
    removed_count = len(participants_df) - len(valid_df)
    if removed_count > 0:
        print(f"⚠️ Filtered {removed_count} participants with null, nan, or empty Encoded IDs")

        # Detailed information on removed participants with circle assignments
        if 'proposed_NEW_circles_id' in participants_df.columns:
            invalid_mask = ~valid_mask
            null_id_with_circle = participants_df[invalid_mask & 
                                              (participants_df['proposed_NEW_circles_id'].notna()) & 
                                              (participants_df['proposed_NEW_circles_id'] != 'UNMATCHED')]

            if len(null_id_with_circle) > 0:
                print(f"  ⚠️ {len(null_id_with_circle)} participants with invalid IDs were assigned to circles:")
                for _, row in null_id_with_circle.iterrows():
                    circle_id = row['proposed_NEW_circles_id']
                    participant_id = row.get('participant_id', 'Unknown')
                    encoded_id = row.get(id_col, 'Unknown')
                    region = row.get('region', 'Unknown')
                    print(f"  - Circle: {circle_id}, encoded_id: {encoded_id}, participant_id: {participant_id}, region: {region}")

    return valid_df

def calculate_matching_statistics(results_df, matched_circles_df):
    """
    Calculate comprehensive matching statistics from results and circles data.
    Uses Results DataFrame as the single source of truth.

    Args:
        results_df: DataFrame with participant results
        matched_circles_df: DataFrame with circle data (can be None)

    Returns:
        Dictionary with standardized statistics
    """
    if results_df is None or results_df.empty:
        return {
            'total_participants': 0,
            'matched_participants': 0,
            'unmatched_participants': 0,
            'match_rate': 0.0,
            'details_matched_count': 0,
            'match_discrepancy': 0,
            'data_source': 'none'
        }

    # Get valid participants (those with non-null Encoded IDs)
    valid_results = get_valid_participants(results_df)

    # Calculate basic statistics from Results DataFrame (single source of truth)
    total_participants = len(valid_results)

    # Count matched participants (not UNMATCHED)
    if 'proposed_NEW_circles_id' in valid_results.columns:
        matched_participants = len(valid_results[valid_results['proposed_NEW_circles_id'] != 'UNMATCHED'])
        unmatched_participants = total_participants - matched_participants
    else:
        matched_participants = 0
        unmatched_participants = total_participants

    # Calculate match rate
    match_rate = (matched_participants / total_participants * 100) if total_participants > 0 else 0

    # Calculate alternative count from circles (for verification only)
    details_matched_count = 0
    match_discrepancy = 0
    data_source = 'results_dataframe'

    if matched_circles_df is not None and not matched_circles_df.empty:
        if 'member_count' in matched_circles_df.columns:
            details_matched_count = matched_circles_df['member_count'].sum()
            match_discrepancy = abs(matched_participants - details_matched_count)
            data_source = 'results_and_circles'
    else:
        # Generate circles data from Results DataFrame for verification
        from modules.demographic_processor import create_circles_dataframe_from_results
        try:
            generated_circles = create_circles_dataframe_from_results(valid_results)
            if not generated_circles.empty and 'member_count' in generated_circles.columns:
                details_matched_count = generated_circles['member_count'].sum()
                match_discrepancy = abs(matched_participants - details_matched_count)
                data_source = 'results_generated_circles'
        except Exception as e:
            print(f"Warning: Could not generate circles from results for verification: {e}")

    # Calculate total circles
    total_circles = 0
    if matched_circles_df is not None and not matched_circles_df.empty:
        # Count circles from matched_circles_df
        if 'circle_id' in matched_circles_df.columns:
            total_circles = len(matched_circles_df['circle_id'].unique())
        else:
            total_circles = len(matched_circles_df)
    elif 'proposed_NEW_circles_id' in valid_results.columns:
        # Count unique circles from Results DataFrame (excluding 'UNMATCHED')
        unique_circles = valid_results[valid_results['proposed_NEW_circles_id'] != 'UNMATCHED']['proposed_NEW_circles_id'].unique()
        total_circles = len(unique_circles)

    return {
        'total_participants': total_participants,
        'matched_participants': matched_participants,
        'unmatched_participants': unmatched_participants,
        'match_rate': match_rate,
        'total_circles': total_circles,
        'details_matched_count': details_matched_count,
        'match_discrepancy': match_discrepancy,
        'data_source': data_source
    }

def ensure_results_has_demographic_categories(results_df):
    """
    Ensure Results DataFrame has all necessary demographic categories.
    This is a key function for making Results DataFrame the single source of truth.

    Args:
        results_df: Results DataFrame

    Returns:
        Results DataFrame with all demographic categories ensured
    """
    if results_df is None or results_df.empty:
        return results_df

    from modules.demographic_processor import ensure_demographic_categories
    return ensure_demographic_categories(results_df)

def get_circles_from_results_or_fallback(results_df, matched_circles_df=None):
    """
    Get circles data using Results DataFrame as primary source, with fallback to matched_circles.
    This implements the single source of truth approach.

    Args:
        results_df: Results DataFrame (primary source)
        matched_circles_df: Matched circles DataFrame (fallback)

    Returns:
        Tuple of (circles_dataframe, data_source_used)
    """
    if results_df is None or results_df.empty:
        if matched_circles_df is not None and not matched_circles_df.empty:
            return matched_circles_df, 'matched_circles_fallback'
        else:
            return pd.DataFrame(), 'none'

    # Try to use matched_circles if available
    if matched_circles_df is not None and not matched_circles_df.empty:
        return matched_circles_df, 'matched_circles'

    # Fallback to generating from Results DataFrame
    from modules.demographic_processor import create_circles_dataframe_from_results
    try:
        generated_circles = create_circles_dataframe_from_results(results_df)
        return generated_circles, 'results_dataframe'
    except Exception as e:
        print(f"Error generating circles from Results DataFrame: {e}")
        return pd.DataFrame(), 'error'

def determine_unmatched_reason(participant, context=None):
    """
    Determine the reason a participant couldn't be matched based on a hierarchical decision tree

    Args:
        participant: Participant data (dict or Series)
        context: Additional context about the optimization state (optional)
            - existing_circles: List of circles with their properties
            - similar_participants: Dict mapping (location, time) to count of compatible participants
            - full_circles: List of circles at maximum capacity (10 members)
            - circles_needing_hosts: List of circles requiring additional hosts
            - compatibility_matrix: Dictionary of participant-circle option compatibility
            - participant_compatible_options: Dictionary mapping participants to their compatible location-time pairs
            - location_time_pairs: List of all possible location-time combinations
            - no_preferences: True if no preferences exist in the region
            - no_location_preferences: True if no location preferences exist in the region
            - no_time_preferences: True if no time preferences exist in the region
            - insufficient_regional_participants: True if the region has too few participants

    Returns:
        Reason code string with the most specific explanation
    """
    # Initialize default context if none provided
    if context is None:
        context = {
            'existing_circles': [],
            'similar_participants': {},
            'full_circles': [],
            'circles_needing_hosts': [],
            'compatibility_matrix': {},
            'participant_compatible_options': {}
        }

    # Get participant ID
    p_id = participant.get('Encoded ID', '')

    # Debug logging for problematic IDs
    debug_mode = context.get('debug_mode', False)
    if p_id in ['66612429591', '71354564939', '65805240273', '76093270642A'] and debug_mode:
        print(f"\n🔍 HIERARCHICAL REASON DETERMINATION for {p_id}:")

    # *** REARRANGED PRIORITY ORDER - PREFERENCES FIRST ***

    # 1. No Preferences Check - most fundamental issue (MOVED TO TOP PRIORITY)
    has_location = bool(participant.get('first_choice_location') or 
                        participant.get('second_choice_location') or 
                        participant.get('third_choice_location'))

    has_time = bool(participant.get('first_choice_time') or 
                    participant.get('second_choice_time') or 
                    participant.get('third_choice_time'))

    if p_id in ['66612429591', '71354564939', '65805240273', '76093270642A', '55467117205'] and debug_mode:
        print(f"  - Has location preferences: {has_location}")
        print(f"  - Has time preferences: {has_time}")

    # Per client request: Use "No location and/or time preferences" whenever either is missing
    # (not just when both are missing)
    if not has_location or not has_time:
        return "No location and/or time preferences"

    # 2. Special context flags from optimizer - only if preferences exist
    if context.get('no_preferences', False):
        return "No location or time preferences"

    if context.get('no_location_preferences', False):
        return "No location preference"

    if context.get('no_time_preferences', False):
        return "No time preference"

    # 3. PER CLIENT REQUEST: NEVER USE "Insufficient participants in region"
    # We've verified all regions have more than 5 participants in total
    # Adding debug logs to help diagnose why this was being triggered
    region = participant.get('Requested_Region', '')
    participant_count = context.get('region_participant_count', {}).get(region, 0)

    if debug_mode:
        print(f"  - Region: {region}")
        print(f"  - Region participant count: {participant_count}")
        print(f"  - Insufficient flag was: {context.get('insufficient_regional_participants', False)}")

    # We'll never use this reason, even if the flag is set

    # 4. No Compatible Options Check
    has_compatible_options = False
    compatible_options = []
    if p_id in context.get('participant_compatible_options', {}):
        compatible_options = context['participant_compatible_options'][p_id]
        has_compatible_options = bool(compatible_options)

    # CRITICAL FIX: If context is empty (greenfield scenarios), manually check compatibility
    if not has_compatible_options and has_location and has_time:
        if debug_mode and p_id in ['66612429591', '71354564939', '65805240273', '76093270642A']:
            print(f"  - Context missing compatibility data, performing manual analysis...")

        # Manual compatibility analysis for greenfield scenarios
        participant_locations = [
            participant.get('first_choice_location', ''),
            participant.get('second_choice_location', ''),
            participant.get('third_choice_location', '')
        ]
        participant_times = [
            participant.get('first_choice_time', ''),
            participant.get('second_choice_time', ''),
            participant.get('third_choice_time', '')
        ]

        # Filter out empty values
        participant_locations = [loc for loc in participant_locations if loc and str(loc).strip()]
        participant_times = [time for time in participant_times if time and str(time).strip()]

        # Create compatible options from cross-product of preferences
        from modules.data_processor import is_time_compatible

        for location in participant_locations:
            for time in participant_times:
                # For greenfield, any location-time combination from preferences is potentially viable
                compatible_options.append((location, time))

        has_compatible_options = bool(compatible_options)

        if debug_mode and p_id in ['66612429591', '71354564939', '65805240273', '76093270642A']:
            print(f"  - Manual analysis found {len(compatible_options)} compatible options: {compatible_options}")

    if p_id in ['66612429591', '71354564939', '65805240273', '76093270642A'] and debug_mode:
        print(f"  - Has compatible options: {has_compatible_options}")
        print(f"  - Compatible options: {compatible_options}")

    if not has_compatible_options:
        return "No compatible location-time combinations"

    # 4.5. Host Validation for Compatible Options (Solution 1 Implementation)
    # Check if compatible location-time combinations have sufficient hosts
    from utils.data_standardization import normalize_host_status

    participant_host_status = normalize_host_status(participant.get('host', ''))
    is_participant_host = participant_host_status in ['ALWAYS', 'SOMETIMES']

    if debug_mode and p_id in ['66612429591', '71354564939', '65805240273', '76093270642A']:
        print(f"  - Participant host status: {participant_host_status} (is_host: {is_participant_host})")

    # Check if any compatible option has sufficient hosts for a new circle
    has_viable_option_with_hosts = False

    for option in compatible_options:
        if isinstance(option, tuple) and len(option) == 2:
            location, time = option

            # Get count of similar participants for this location-time
            similar_count = context.get('similar_participants', {}).get((location, time), 0)

            # Get host count for this location-time combination
            host_count = context.get('host_counts', {}).get((location, time), 0)

            # GREENFIELD FIX: If no context data available, manually calculate host availability
            if similar_count == 0 and host_count == 0:
                # This indicates we're in a greenfield scenario without context data
                # In this case, if the participant is a host, we have at least 1 host available
                # If not, we need to assume no hosts are available for this specific check
                if is_participant_host:
                    effective_host_count = 1  # This participant can host
                    # For greenfield, assume other compatible participants exist if prefs are common
                    effective_similar_count = 5  # Assume minimum viable circle size
                else:
                    effective_host_count = 0  # No hosts available
                    effective_similar_count = 5  # Still assume participants exist, but no hosts
            else:
                # Use context data if available
                effective_host_count = host_count + (1 if is_participant_host else 0)
                effective_similar_count = similar_count

            if debug_mode and p_id in ['66612429591', '71354564939', '65805240273', '76093270642A']:
                print(f"  - Option ({location}, {time}): {effective_similar_count} participants, {effective_host_count} hosts")

            # Check if this option could form a viable circle (min 5 participants, min 1 host)
            if effective_similar_count >= 4 and effective_host_count >= 1:  # 4 others + this participant = 5 total
                has_viable_option_with_hosts = True
                break

    # If compatible options exist but none have sufficient hosts, return specific message
    if not has_viable_option_with_hosts:
        if debug_mode and p_id in ['66612429591', '71354564939', '65805240273', '76093270642A']:
            print(f"  - No viable options with hosts found")
        return "Insufficient hosts available"

    # 5. Very Limited Options Check - if there are very few compatible options
    # Only apply this if the participant has at least some preferences
    if p_id in context.get('participant_compatible_count', {}):
        option_count = context['participant_compatible_count'][p_id]

        if p_id in ['66612429591', '71354564939', '65805240273', '76093270642A'] and debug_mode:
            print(f"  - Compatible option count: {option_count}")

        if option_count < 2:
            return "Very limited compatible options"

    # Note: Removed waitlist check as waitlisted participants should be treated the same as others

    # Get participant locations and times
    participant_locations = [
        participant.get('first_choice_location', ''),
        participant.get('second_choice_location', ''),
        participant.get('third_choice_location', '')
    ]

    participant_times = [
        participant.get('first_choice_time', ''),
        participant.get('second_choice_time', ''),
        participant.get('third_choice_time', '')
    ]

    # Filter out empty values
    participant_locations = [loc for loc in participant_locations if loc]
    participant_times = [time for time in participant_times if time]

    # 6. Location Match Check
    # Check if any compatible locations have enough participants
    location_matches = []
    for location in participant_locations:
        # Check if this location has at least one potential circle
        has_potential = False
        for loc_time_key, count in context.get('similar_participants', {}).items():
            loc, _ = loc_time_key  # Unpack the tuple
            if loc == location and count >= 4:  # Need at least 4 others (5 total with this participant)
                has_potential = True
                location_matches.append(location)
                break

    if not location_matches:
        return "No location with sufficient participants"

    # 7. Time Match at Location Check
    # Check if any compatible location-time combinations have enough participants
    time_location_matches = []
    for location in location_matches:
        for time in participant_times:
            loc_time_key = (location, time)
            if context.get('similar_participants', {}).get(loc_time_key, 0) >= 4:
                time_location_matches.append(loc_time_key)

    if not time_location_matches:
        return "No time match with sufficient participants"

    # 8. Host Requirement Check
    is_host = False
    host_value = str(participant.get('host', '')).lower()
    if host_value in ['always', 'always host', 'sometimes', 'sometimes host']:
        is_host = True

    # Check for in-person circles needing hosts
    needs_host_locations = set()
    for circle in context.get('circles_needing_hosts', []):
        if circle.get('subregion') in participant_locations:
            needs_host_locations.add(circle.get('subregion'))

    if needs_host_locations and not is_host:
        location_strings = ', '.join(needs_host_locations)
        return f"Host requirement not met at {location_strings}"

    # 9. Circle Capacity Check
    all_compatible_circles_full = True
    for location, time in time_location_matches:
        # Check if any circles at this location/time are not full
        for circle in context.get('existing_circles', []):
            if (circle.get('subregion') == location and 
                circle.get('meeting_time') == time and
                circle.get('circle_id') not in context.get('full_circles', [])):
                all_compatible_circles_full = False
                break

        # Also check if we could potentially create a new circle here
        if context.get('similar_participants', {}).get((location, time), 0) >= 5:  # Minimum circle size
            all_compatible_circles_full = False
            break

    if all_compatible_circles_full:
        return "All compatible circles at capacity"

    # 10. Host Capacity for New Circles Check
    if not is_host:
        # Check if there are enough hosts among similar participants for each location-time pair
        hosts_available = False
        for location, time in time_location_matches:
            similar_count = context.get('similar_participants', {}).get((location, time), 0)
            host_count = context.get('host_counts', {}).get((location, time), 0)

            if similar_count >= 4 and host_count > 0:
                hosts_available = True
                break

        if not hosts_available:
            return "Insufficient hosts for compatible options"

    # 11. Default Reason - per client request, use a simpler message
    # This is our default if all other checks pass but participant is still unmatched
    return "Tool unable to find a match"

def is_valid_member_id(member_id):
    """
    Check if a member ID is valid (not nan, None, empty, etc.)
    
    Args:
        member_id: The member ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if member_id is None or pd.isna(member_id):
        return False
    
    # Convert to string for checking
    member_id_str = str(member_id).strip()
    
    # Check for various invalid formats
    if (member_id_str == '' or 
        member_id_str.lower() == 'nan' or 
        member_id_str == 'None' or
        member_id_str == 'null'):
        return False
    
    return True

def clean_member_list(member_list):
    """
    Clean a list of member IDs by removing invalid entries
    
    Args:
        member_list: List of member IDs (can be list, string, or other)
        
    Returns:
        List of valid member IDs
    """
    if not member_list:
        return []
    
    # Handle string representation of lists
    if isinstance(member_list, str):
        try:
            if member_list.startswith('[') and member_list.endswith(']'):
                member_list = eval(member_list)
            else:
                return [member_list] if is_valid_member_id(member_list) else []
        except:
            return [member_list] if is_valid_member_id(member_list) else []
    
    # Handle actual lists
    if isinstance(member_list, list):
        return [member_id for member_id in member_list if is_valid_member_id(member_id)]
    
    # Handle single values
    return [member_list] if is_valid_member_id(member_list) else []

def normalize_string(s):
    """
    Apply basic normalization to a string

    Args:
        s: String to normalize

    Returns:
        Normalized string
    """
    if pd.isna(s) or not s:
        return ''

    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)  # Replace multiple spaces with single space

    return s