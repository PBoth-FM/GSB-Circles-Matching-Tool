"""
Post-processing utilities to fix invalid circle IDs in the final results.
This module handles the correction of "IP-UNKNOWN" circle patterns by properly
splitting them into correctly named circles based on participant data.
"""

import pandas as pd
import re
from typing import Dict, List, Tuple, Set
from utils.normalization import determine_region_code


def detect_unknown_circles(results_df: pd.DataFrame) -> List[str]:
    """
    Detect any circle IDs containing 'UNKNOWN' patterns.
    
    Args:
        results_df: DataFrame with participant results
        
    Returns:
        List of problematic circle IDs
    """
    if 'proposed_NEW_circles_id' not in results_df.columns:
        return []
    
    # Find all circle IDs containing "UNKNOWN"
    unknown_mask = results_df['proposed_NEW_circles_id'].str.contains('UNKNOWN', na=False)
    unknown_ids = results_df[unknown_mask]['proposed_NEW_circles_id'].unique().tolist()
    
    if unknown_ids:
        total_participants = results_df[unknown_mask].shape[0]
        print(f"🔍 DETECTION: Found {len(unknown_ids)} invalid 'UNKNOWN' circles affecting {total_participants} participants")
        for circle_id in unknown_ids:
            count = results_df[results_df['proposed_NEW_circles_id'] == circle_id].shape[0]
            print(f"  - {circle_id}: {count} participants")
    
    return unknown_ids


def analyze_unknown_groups(results_df: pd.DataFrame, unknown_circle_ids: List[str]) -> Dict[str, List[pd.DataFrame]]:
    """
    Analyze each unknown circle to identify distinct groups that should be separate circles.
    
    Args:
        results_df: DataFrame with participant results
        unknown_circle_ids: List of problematic circle IDs
        
    Returns:
        Dictionary mapping original circle ID to list of participant groups
    """
    grouped_data = {}
    
    for circle_id in unknown_circle_ids:
        # Get all participants with this circle ID
        participants = results_df[results_df['proposed_NEW_circles_id'] == circle_id].copy()
        
        if participants.empty:
            continue
            
        # Group by region and subregion to identify distinct circles
        group_columns = []
        if 'Derived_Region' in participants.columns:
            group_columns.append('Derived_Region')
        if 'proposed_NEW_Subregion' in participants.columns:
            group_columns.append('proposed_NEW_Subregion')
        
        if not group_columns:
            print(f"⚠️ WARNING: Cannot group {circle_id} - missing region/subregion columns")
            continue
        
        # Group participants by region and subregion
        groups = []
        for group_key, group_df in participants.groupby(group_columns):
            groups.append(group_df)
            
        grouped_data[circle_id] = groups
        
        # Log the analysis
        print(f"📊 ANALYSIS: Circle {circle_id} contains {len(groups)} distinct groups:")
        for i, group in enumerate(groups):
            region = group['Derived_Region'].iloc[0] if 'Derived_Region' in group.columns else 'Unknown'
            subregion = group['proposed_NEW_Subregion'].iloc[0] if 'proposed_NEW_Subregion' in group.columns else 'Unknown'
            print(f"  Group {i+1}: {region} / {subregion} ({len(group)} participants)")
    
    return grouped_data


def generate_proper_circle_name(group_df: pd.DataFrame) -> str:
    """
    Generate a proper circle name for a group of participants.
    
    Args:
        group_df: DataFrame containing participants from the same actual circle
        
    Returns:
        Properly formatted circle ID
    """
    # Extract region and subregion information
    region = group_df['Derived_Region'].iloc[0] if 'Derived_Region' in group_df.columns else 'Virtual-Only APAC+EMEA'
    subregion = group_df['proposed_NEW_Subregion'].iloc[0] if 'proposed_NEW_Subregion' in group_df.columns else 'GMT'
    
    # Determine the proper region code
    region_code = determine_region_code(region, subregion)
    
    # Generate the base circle name
    if region_code and region_code != 'UNKNOWN':
        # For virtual circles, include timezone information
        if 'GMT+3' in str(subregion):
            base_name = f"VO-{region_code}-GMT+3-NEW"
        elif 'GMT+8' in str(subregion):
            base_name = f"VO-{region_code}-GMT+8-NEW"
        elif 'GMT-6' in str(subregion) or 'Central Standard Time' in str(subregion):
            base_name = f"VO-{region_code}-GMT-6-NEW"
        elif 'GMT-5' in str(subregion) or 'Eastern Standard Time' in str(subregion):
            base_name = f"VO-{region_code}-GMT-5-NEW"
        elif 'GMT-7' in str(subregion) or 'Mountain Standard Time' in str(subregion):
            base_name = f"VO-{region_code}-GMT-7-NEW"
        elif 'GMT-8' in str(subregion) or 'Pacific Standard Time' in str(subregion):
            base_name = f"VO-{region_code}-GMT-8-NEW"
        else:
            base_name = f"VO-{region_code}-GMT-NEW"
    else:
        # Fallback for unknown regions
        base_name = "VO-AE-GMT-NEW"
    
    return base_name


def find_next_available_number(base_name: str, existing_ids: Set[str]) -> str:
    """
    Find the next available number for a circle name pattern.
    
    Args:
        base_name: Base circle name (e.g., "VO-AE-GMT-NEW")
        existing_ids: Set of existing circle IDs
        
    Returns:
        Complete circle ID with available number
    """
    # Find all existing circles with this base pattern
    pattern = rf"{re.escape(base_name)}-(\d+)"
    existing_numbers = []
    
    for circle_id in existing_ids:
        match = re.match(pattern, circle_id)
        if match:
            existing_numbers.append(int(match.group(1)))
    
    # Find the next available number
    if not existing_numbers:
        next_number = 1
    else:
        next_number = max(existing_numbers) + 1
    
    return f"{base_name}-{next_number:02d}"


def update_dataframe_with_corrected_ids(results_df: pd.DataFrame, corrections: Dict[Tuple[int, ...], str]) -> pd.DataFrame:
    """
    Update the dataframe with corrected circle IDs.
    
    Args:
        results_df: Original DataFrame
        corrections: Dictionary mapping participant indices to new circle IDs
        
    Returns:
        Updated DataFrame with corrected circle IDs
    """
    results_df = results_df.copy()
    
    for indices, new_circle_id in corrections.items():
        # Update the circle ID for these participants
        results_df.loc[list(indices), 'proposed_NEW_circles_id'] = new_circle_id
        
        # Also update any other related columns that might reference the circle ID
        if 'circles_id' in results_df.columns:
            results_df.loc[list(indices), 'circles_id'] = new_circle_id
    
    return results_df


def fix_unknown_circle_ids(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function to fix all unknown circle IDs in the results dataframe.
    
    Args:
        results_df: DataFrame with participant results
        
    Returns:
        DataFrame with corrected circle IDs
    """
    print("\n🔧 POST-PROCESSING: Fixing invalid circle IDs")
    
    # Step 1: Detect unknown circles
    unknown_ids = detect_unknown_circles(results_df)
    if not unknown_ids:
        print("✅ No invalid circle IDs found - no corrections needed")
        return results_df
    
    # Step 2: Analyze groups within each unknown circle
    grouped_data = analyze_unknown_groups(results_df, unknown_ids)
    
    # Step 3: Generate corrections
    existing_ids = set(results_df['proposed_NEW_circles_id'].dropna())
    corrections = {}
    
    for original_id, groups in grouped_data.items():
        print(f"\n🔧 CORRECTING: Processing {original_id}")
        
        for i, group in enumerate(groups):
            # Generate proper name for this group
            base_name = generate_proper_circle_name(group)
            final_name = find_next_available_number(base_name, existing_ids)
            
            # Store correction mapping using participant indices
            participant_indices = tuple(group.index)
            corrections[participant_indices] = final_name
            existing_ids.add(final_name)
            
            # Log the transformation
            region = group['Derived_Region'].iloc[0] if 'Derived_Region' in group.columns else 'Unknown'
            subregion = group['proposed_NEW_Subregion'].iloc[0] if 'proposed_NEW_Subregion' in group.columns else 'Unknown'
            print(f"  ✅ Group {i+1}: {original_id} → {final_name}")
            print(f"     Region: {region}, Subregion: {subregion}, Participants: {len(group)}")
    
    # Step 4: Apply corrections
    if corrections:
        results_df = update_dataframe_with_corrected_ids(results_df, corrections)
        
        # Verify no unknown patterns remain
        remaining_unknown = detect_unknown_circles(results_df)
        if remaining_unknown:
            print(f"⚠️ WARNING: {len(remaining_unknown)} unknown circles still remain after correction")
        else:
            print(f"✅ SUCCESS: Fixed all invalid circle IDs, created {len(corrections)} properly named circles")
    
    return results_df


def has_unknown_circles(results_df: pd.DataFrame) -> bool:
    """
    Quick check if the dataframe contains any unknown circle patterns.
    
    Args:
        results_df: DataFrame to check
        
    Returns:
        True if unknown patterns are found
    """
    if 'proposed_NEW_circles_id' not in results_df.columns:
        return False
    
    return results_df['proposed_NEW_circles_id'].str.contains('UNKNOWN', na=False).any()