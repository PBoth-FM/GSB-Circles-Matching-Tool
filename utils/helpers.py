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
        'proposed_NEW_co_leader'
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
    
    # All other columns (except name/email columns that we'll place later)
    remaining_columns = [col for col in output_df.columns 
                        if col not in ordered_columns 
                        and col not in name_email_columns]
    
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
        
    Returns:
        Reason code string with the most specific explanation
    """
    # Initialize default context if none provided
    if context is None:
        context = {
            'existing_circles': [],
            'similar_participants': {},
            'full_circles': [],
            'circles_needing_hosts': []
        }
    
    # 1. No Preferences Check - most fundamental issue
    if (not participant.get('first_choice_location') and 
        not participant.get('second_choice_location') and 
        not participant.get('third_choice_location') and
        not participant.get('first_choice_time') and 
        not participant.get('second_choice_time') and 
        not participant.get('third_choice_time')):
        return "No location or time preferences"
    
    # 2. Special Status Check
    status = participant.get('Raw_Status', participant.get('Status', ''))
    if isinstance(status, str) and 'WAITLIST' in status.upper() and 'LOW PRIORITY' in status.upper():
        return "Waitlist - low priority"
    
    # 3. Partial Preference Checks
    # No Location Preference
    if (not participant.get('first_choice_location') and 
        not participant.get('second_choice_location') and 
        not participant.get('third_choice_location')):
        return "No location preference"
    
    # No Time Preference
    if (not participant.get('first_choice_time') and 
        not participant.get('second_choice_time') and 
        not participant.get('third_choice_time')):
        return "No time preference"
    
    # 4. Location Match Check
    location_matches = False
    participant_locations = [
        participant.get('first_choice_location', ''),
        participant.get('second_choice_location', ''),
        participant.get('third_choice_location', '')
    ]
    
    # Remove empty values
    participant_locations = [loc for loc in participant_locations if loc]
    
    # Check if any of participant's locations match existing circles or have enough similar participants
    for location in participant_locations:
        # Check existing circles
        for circle in context.get('existing_circles', []):
            if circle.get('subregion') == location:
                location_matches = True
                break
                
        # Check for similar participants at this location
        for loc_time_key, count in context.get('similar_participants', {}).items():
            loc, _ = loc_time_key  # Unpack the tuple
            if loc == location and count >= 4:  # Need at least 4 others (5 total with this participant)
                location_matches = True
                break
                
        if location_matches:
            break
    
    if not location_matches:
        return "No location matches"
    
    # 5. Time Match at Location Check
    time_match_at_location = False
    participant_times = [
        participant.get('first_choice_time', ''),
        participant.get('second_choice_time', ''),
        participant.get('third_choice_time', '')
    ]
    
    # Remove empty values
    participant_times = [time for time in participant_times if time]
    
    # Check for each location-time combination
    for location in participant_locations:
        for time in participant_times:
            # Check existing circles
            for circle in context.get('existing_circles', []):
                if circle.get('subregion') == location and circle.get('meeting_time') == time:
                    time_match_at_location = True
                    break
                    
            # Check for similar participants at this location and time
            loc_time_key = (location, time)
            if context.get('similar_participants', {}).get(loc_time_key, 0) >= 4:
                time_match_at_location = True
                break
                
            if time_match_at_location:
                break
        
        if time_match_at_location:
            break
    
    if not time_match_at_location:
        return "No time match at this location"
    
    # 6. Host Requirement Check
    is_host = participant.get('host', '').lower() in ['always', 'always host', 'sometimes', 'sometimes host']
    needs_in_person = any(loc_time_key[0].startswith('IP-') for loc_time_key in context.get('similar_participants', {}))
    
    if needs_in_person and not is_host and context.get('circles_needing_hosts', []):
        for circle in context.get('circles_needing_hosts', []):
            if circle.get('subregion') in participant_locations:
                return "Host requirement not met"
    
    # 7. Circle Capacity Check
    all_circles_full = True
    for circle in context.get('existing_circles', []):
        if (circle.get('subregion') in participant_locations and 
            circle.get('meeting_time') in participant_times and
            circle.get('circle_id') not in context.get('full_circles', [])):
            all_circles_full = False
            break
    
    if all_circles_full and context.get('existing_circles', []):
        return "Compatible circles are full"
    
    # 8. Similar Participants Check
    has_enough_similar = False
    for location in participant_locations:
        for time in participant_times:
            loc_time_key = (location, time)
            if context.get('similar_participants', {}).get(loc_time_key, 0) >= 4:
                has_enough_similar = True
                break
        if has_enough_similar:
            break
    
    if not has_enough_similar:
        return "Insufficient similar participants"
    
    # 9. Host Capacity for New Circles Check
    if needs_in_person and not is_host:
        # Check if there are enough hosts among similar participants
        hosts_available = False
        for location in participant_locations:
            for time in participant_times:
                loc_time_key = (location, time)
                similar_count = context.get('similar_participants', {}).get(loc_time_key, 0)
                host_count = context.get('host_counts', {}).get(loc_time_key, 0)
                
                if similar_count >= 4 and host_count > 0:
                    hosts_available = True
                    break
            if hosts_available:
                break
        
        if not hosts_available:
            return "Insufficient hosts for new circle"
    
    # 10. Algorithm Optimization Check
    # This is our default if all other checks pass but participant is still unmatched
    return "Algorithm couldn't optimize placement"

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
