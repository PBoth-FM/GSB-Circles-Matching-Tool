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
    Generate a downloadable link for a DataFrame
    
    Args:
        df: Pandas DataFrame to convert
        
    Returns:
        CSV data as string
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def generate_circle_id(region, subregion, index):
    """
    Generate a circle ID following the naming convention
    
    Args:
        region: Region name
        subregion: Subregion name
        index: Circle index
        
    Returns:
        Circle ID string
    """
    # Format: IP-NEW-{RegionCode}-{index}
    # Example: IP-NEW-SFL-01
    
    # Normalize region and generate code
    region_code = ''.join([word[0] for word in region.split() if word.lower() not in ['of', 'the', 'and']])
    
    # Make sure it's uppercase
    region_code = region_code.upper()
    
    # Format the index as 2-digit number
    index_str = str(index).zfill(2)
    
    return f"IP-NEW-{region_code}-{index_str}"

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

def determine_unmatched_reason(participant):
    """
    Determine the reason a participant couldn't be matched
    
    Args:
        participant: Participant data (dict or Series)
        
    Returns:
        Reason code string
    """
    # Check for no preferences at all
    if (not participant.get('first_choice_location') and 
        not participant.get('second_choice_location') and 
        not participant.get('third_choice_location') and
        not participant.get('first_choice_time') and 
        not participant.get('second_choice_time') and 
        not participant.get('third_choice_time')):
        return "No preferences"
    
    # Check for no location preferences
    if (not participant.get('first_choice_location') and 
        not participant.get('second_choice_location') and 
        not participant.get('third_choice_location')):
        return "No location preference"
    
    # Check for no time preferences
    if (not participant.get('first_choice_time') and 
        not participant.get('second_choice_time') and 
        not participant.get('third_choice_time')):
        return "No time preference"
    
    # Default to optimization trade-off
    return "Optimization trade-off"

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
