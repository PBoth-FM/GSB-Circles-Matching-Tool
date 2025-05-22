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
    
    # Define the column order according to specifications
    ordered_columns = []
    
    # First column should be Status
    if 'Status' in output_df.columns:
        ordered_columns.append('Status')
    
    # Keep Raw_Status next to Status if available
    if 'Raw_Status' in output_df.columns:
        ordered_columns.append('Raw_Status')
    
    # Next column should be Encoded ID
    if 'Encoded ID' in output_df.columns:
        ordered_columns.append('Encoded ID')
    
    # Next come the specified columns in order
    priority_columns = [
        'proposed_NEW_circles_id',
        'unmatched_reason',
        'proposed_NEW_Subregion',
        'proposed_NEW_DayTime',
        'proposed_NEW_host',
        'proposed_NEW_co_leader',
        'max_additions'  # Added to include the max_additions data
    ]
    
    for col in priority_columns:
        if col in output_df.columns:
            ordered_columns.append(col)
    
    # Identify name and email columns to place them in the right spot
    name_email_columns = []
    if 'Last (Family) Name' in output_df.columns:
        name_email_columns.append('Last (Family) Name')
    if 'First (Given) Name' in output_df.columns:
        name_email_columns.append('First (Given) Name')
    if 'Preferred Email' in output_df.columns:
        name_email_columns.append('Preferred Email')
    
    # Find GSB Class and Class Vintage columns to place them together
    gsb_class_column = None
    gsb_vintage_column = None
    
    # Look for GSB Class column with improved detection logic
    # First try for the exact column name that we know is used in the input data
    if 'GSB Class Year' in output_df.columns:
        gsb_class_column = 'GSB Class Year'
        print(f"Found exact GSB Class Year column")
    else:
        # Fallback to case-insensitive search with better pattern matching
        for col in output_df.columns:
            if any(term in col.lower().replace(" ", "") for term in ['gsbclass', 'gsb class']):
                gsb_class_column = col
                print(f"Found GSB Class column via pattern match: '{col}'")
                break
    
    # Class Vintage column - ensure it's included
    if 'Class_Vintage' in output_df.columns:
        gsb_vintage_column = 'Class_Vintage'
        print(f"Found Class_Vintage column")
    
    # Debug to verify column inclusion
    if gsb_class_column:
        print(f"Will include GSB Class column: '{gsb_class_column}'")
        # Check if it has data
        non_null_values = output_df[gsb_class_column].notna().sum()
        print(f"- GSB Class column has {non_null_values} non-null values out of {len(output_df)}")
    
    if gsb_vintage_column:
        print(f"Will include Class Vintage column: '{gsb_vintage_column}'")
        # Check if it has data
        non_null_values = output_df[gsb_vintage_column].notna().sum()
        print(f"- Class Vintage column has {non_null_values} non-null values out of {len(output_df)}")
    
    # All other columns (except name/email columns and GSB class columns that we'll place later)
    remaining_columns = [col for col in output_df.columns 
                        if col not in ordered_columns 
                        and col not in name_email_columns
                        and col != gsb_class_column
                        and col != gsb_vintage_column]
    
    # Add remaining columns alphabetically for consistency
    ordered_columns.extend(sorted(remaining_columns))
    
    # Now insert the name/email columns just before the Preferred Email
    if 'Preferred Email' in output_df.columns:
        email_index = ordered_columns.index('Preferred Email') if 'Preferred Email' in ordered_columns else len(ordered_columns)
        
        # If Found, remove Preferred Email from ordered_columns first
        if 'Preferred Email' in ordered_columns:
            ordered_columns.remove('Preferred Email')
        
        # Insert name columns followed by Preferred Email at the right position
        for col in name_email_columns:
            if col != 'Preferred Email' and col in output_df.columns:
                ordered_columns.insert(email_index, col)
                email_index += 1
                
        # Add Preferred Email back at the right position
        if 'Preferred Email' in output_df.columns:
            ordered_columns.insert(email_index, 'Preferred Email')
    
    # Add GSB Class and Class Vintage columns in the right order if they exist
    if gsb_class_column and gsb_class_column not in ordered_columns:
        # Add GSB Class column first
        ordered_columns.append(gsb_class_column)
        
        # Add Class Vintage right after GSB Class column if it exists
        if gsb_vintage_column and gsb_vintage_column not in ordered_columns:
            gsb_class_index = ordered_columns.index(gsb_class_column)
            ordered_columns.insert(gsb_class_index + 1, gsb_vintage_column)
    elif gsb_vintage_column and gsb_vintage_column not in ordered_columns:
        # If we only have Class Vintage but no GSB Class, just add it
        ordered_columns.append(gsb_vintage_column)
    
    # Make sure we haven't lost any columns
    for col in output_df.columns:
        if col not in ordered_columns and not col.startswith('Unnamed:'):
            ordered_columns.append(col)
    
    # Create a new DataFrame with only the columns that exist
    final_columns = [col for col in ordered_columns if col in output_df.columns]
    final_df = output_df[final_columns]
    
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
    else:
        # For in-person circles, use the standard region code
        region_code = get_region_code(region)
        
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
    
    # Filter out null IDs and convert to string to handle numeric IDs properly
    valid_df = participants_df[participants_df[id_col].notna() & 
                           (participants_df[id_col].astype(str) != 'None') &
                           (participants_df[id_col].astype(str) != '')]
    
    # Log the filtering process for debugging
    removed_count = len(participants_df) - len(valid_df)
    if removed_count > 0:
        print(f"⚠️ Filtered {removed_count} participants with null or empty Encoded IDs")
        
        # Detailed information on removed participants with circle assignments
        if 'proposed_NEW_circles_id' in participants_df.columns:
            null_id_mask = participants_df[id_col].isna() | (participants_df[id_col].astype(str) == 'None') | (participants_df[id_col].astype(str) == '')
            null_id_with_circle = participants_df[null_id_mask & 
                                              (participants_df['proposed_NEW_circles_id'].notna()) & 
                                              (participants_df['proposed_NEW_circles_id'] != 'UNMATCHED')]
            
            if len(null_id_with_circle) > 0:
                print(f"  ⚠️ {len(null_id_with_circle)} participants with null IDs were assigned to circles:")
                for _, row in null_id_with_circle.iterrows():
                    circle_id = row['proposed_NEW_circles_id']
                    participant_id = row.get('participant_id', 'Unknown')
                    region = row.get('region', 'Unknown')
                    print(f"  - Circle: {circle_id}, participant_id: {participant_id}, region: {region}")
    
    return valid_df

def calculate_matching_statistics(results_df, circles_df=None):
    """
    Calculate standardized statistics for the matching process to ensure consistency
    across all parts of the application
    
    Args:
        results_df: DataFrame with participant results including circle assignments
        circles_df: Optional DataFrame with circle data including member counts
        
    Returns:
        Dictionary with standardized statistics:
            - total_participants: Total number of participants
            - matched_participants: Number of participants successfully matched
            - unmatched_participants: Number of participants not matched
            - match_rate: Percentage of participants matched
            - total_circles: Total number of circles created
            - continuing_circles: Number of existing/continuing circles
            - new_circles: Number of newly created circles
            - avg_circle_size: Average circle size
            - adjusted_statistics: Statistics after excluding test circles
    """
    stats = {}
    
    # Initialize with default values
    stats['total_participants'] = 0
    stats['matched_participants'] = 0 
    stats['unmatched_participants'] = 0
    stats['match_rate'] = 0.0
    stats['total_circles'] = 0
    stats['continuing_circles'] = 0
    stats['new_circles'] = 0
    stats['avg_circle_size'] = 0.0
    
    # Handle case where results_df is None or empty
    if results_df is None or len(results_df) == 0:
        return stats
    
    # Filter out participants with null Encoded IDs using our new helper function
    # This ensures consistent counting across all statistics
    valid_results_df = get_valid_participants(results_df)
    
    # Store original count for debugging
    original_count = len(results_df)
    
    # Calculate participant statistics from filtered results DataFrame
    stats['total_participants'] = len(valid_results_df)
    stats['filtered_participants'] = original_count - len(valid_results_df)
    
    if 'proposed_NEW_circles_id' in valid_results_df.columns:
        # Count matched participants (using the Match page method)
        valid_circle_mask = (valid_results_df['proposed_NEW_circles_id'].notna()) & (valid_results_df['proposed_NEW_circles_id'] != 'UNMATCHED')
        stats['matched_participants'] = len(valid_results_df[valid_circle_mask])
        
        # Count unmatched participants
        stats['unmatched_participants'] = len(valid_results_df[valid_results_df['proposed_NEW_circles_id'] == 'UNMATCHED'])
        
        # Calculate match rate
        if stats['total_participants'] > 0:
            stats['match_rate'] = (stats['matched_participants'] / stats['total_participants']) * 100
    
    # Calculate circle statistics if circle data is provided
    if circles_df is not None and len(circles_df) > 0:
        # Total circles
        stats['total_circles'] = len(circles_df)
        
        # Count continuing circles (not starting with IP-NEW)
        if 'circle_id' in circles_df.columns:
            new_circle_mask = circles_df['circle_id'].str.contains('IP-NEW-', na=False)
            stats['new_circles'] = new_circle_mask.sum()
            stats['continuing_circles'] = len(circles_df) - stats['new_circles']
        
        # Calculate average circle size
        if 'member_count' in circles_df.columns:
            stats['avg_circle_size'] = circles_df['member_count'].mean()
            
            # Calculate alternative matched count (using Details page method)
            details_matched_count = circles_df['member_count'].sum()
            stats['details_matched_count'] = details_matched_count
            
            # Record the discrepancy between methods
            stats['match_discrepancy'] = details_matched_count - stats['matched_participants']
            
            # Check for test circles
            test_circles = []
            test_patterns = ['IP-TEST', 'IP-SIN-01', 'IP-LON-04', 'IP-HOU-02']
            
            if 'circle_id' in circles_df.columns:
                for pattern in test_patterns:
                    test_mask = circles_df['circle_id'].str.contains(pattern, na=False)
                    test_circles.extend(circles_df[test_mask]['circle_id'].tolist())
                
                # If test circles exist, calculate adjusted statistics
                if test_circles:
                    stats['test_circles'] = test_circles
                    
                    # Calculate participants in test circles
                    test_circle_participants = 0
                    if 'member_count' in circles_df.columns:
                        test_circle_participants = circles_df[circles_df['circle_id'].isin(test_circles)]['member_count'].sum()
                    
                    # Create adjusted statistics
                    stats['adjusted_statistics'] = {
                        'test_circle_participants': test_circle_participants,
                        'adjusted_matched_participants': stats['matched_participants'] - test_circle_participants,
                        'adjusted_match_rate': ((stats['matched_participants'] - test_circle_participants) / 
                                              stats['total_participants']) * 100 if stats['total_participants'] > 0 else 0.0,
                        'adjusted_total_circles': stats['total_circles'] - len(test_circles),
                    }
    
    return stats

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
    if p_id in context.get('participant_compatible_options', {}):
        has_compatible_options = bool(context['participant_compatible_options'][p_id])
    
    if p_id in ['66612429591', '71354564939', '65805240273', '76093270642A'] and debug_mode:
        print(f"  - Has compatible options: {has_compatible_options}")
    
    if not has_compatible_options:
        return "No compatible location-time combinations"
    
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
