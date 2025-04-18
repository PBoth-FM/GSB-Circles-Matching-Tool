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
    
    # Only keep columns that don't start with "Unnamed:"
    filtered_columns = [col for col in output_df.columns if not col.startswith('Unnamed:')]
    output_df = output_df[filtered_columns]
    
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
        Circle ID string
    """
    # Import here to avoid circular imports
    from utils.normalization import get_region_code
    
    # Get the standardized region code
    region_code = get_region_code(region)
    
    # Format the index as 2-digit number
    index_str = str(index).zfill(2)
    
    # Format: IP-NEW-{RegionCode}-{index} for new circles
    # For existing circles, the format is just IP-{RegionCode}-{index}
    if is_new:
        return f"IP-NEW-{region_code}-{index_str}"
    else:
        return f"IP-{region_code}-{index_str}"

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
