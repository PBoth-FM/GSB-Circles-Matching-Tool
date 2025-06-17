"""
Co-Leader Assignment Module

This module implements the business logic for assigning co-leaders to circles
after the optimization algorithm has completed circle assignments.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def find_co_leader_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Find the actual column names for co-leader related fields in the dataframe.
    Handles variations in column naming between input data and processed data.
    
    Args:
        df: DataFrame to search for co-leader columns
        
    Returns:
        Dictionary mapping standard names to actual column names
    """
    column_map = {}
    
    # Map for Current Co-Leader status
    current_co_leader_variations = [
        'Current Co-Leader?', 'Current_Co_Leader', 'Current Co-Leader', 
        'Co-Leader Status', 'Current Circle Co-Leader'
    ]
    for col in current_co_leader_variations:
        if col in df.columns:
            column_map['current_co_leader'] = col
            break
    
    # Map for Co-Leader Response about 2025
    co_leader_response_variations = [
        'Co-Leader Response:  CL in 2025?', 'Co-Leader Response: CL in 2025?',
        'Co-Leader Response CL in 2025', 'CL Response 2025'
    ]
    for col in co_leader_response_variations:
        if col in df.columns:
            column_map['co_leader_response_2025'] = col
            break
    
    # Map for Non-CLs Volunteering to Co-Lead
    volunteering_variations = [
        '(Non CLs) Volunteering to Co-Lead?', 'Volunteering to Co-Lead?',
        'Volunteering to Co-Lead', 'Non CLs Volunteering to Co-Lead'
    ]
    for col in volunteering_variations:
        if col in df.columns:
            column_map['volunteering_to_co_lead'] = col
            break
    
    return column_map


def assign_co_leaders(results_df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
    """
    Assign co-leaders to each circle based on the specified business rules.
    
    Business Rules:
    1. For CURRENT-CONTINUING participants who are current co-leaders:
       - If "Co-Leader Response: CL in 2025?" is "No" → proposed_NEW_Coleader = "No"
       - Otherwise → proposed_NEW_Coleader = "Yes"
    
    2. For all other participants (non-CURRENT-CONTINUING):
       - Always → proposed_NEW_Coleader = "Yes" (regardless of volunteering response)
    
    3. Each circle must have at least 2 co-leaders:
       - If fewer than 2, set ALL participants in that circle as co-leaders
    
    Args:
        results_df: DataFrame with participant assignments and circle IDs
        debug_mode: Whether to print debug information
        
    Returns:
        DataFrame with added 'proposed_NEW_Coleader' column
    """
    if debug_mode:
        print("\n🎯 CO-LEADER ASSIGNMENT: Starting assignment process")
        print(f"  Input DataFrame shape: {results_df.shape}")
        print(f"  Columns available: {results_df.columns.tolist()}")
    
    # Create a copy to avoid modifying the original
    df = results_df.copy()
    
    # Find the actual column names in this dataframe
    column_map = find_co_leader_columns(df)
    
    if debug_mode:
        print(f"  Column mapping found: {column_map}")
    
    # Initialize the co-leader column
    df['proposed_NEW_Coleader'] = 'No'
    
    # Get counts for debugging
    total_participants = len(df)
    matched_participants = len(df[df['proposed_NEW_circles_id'] != 'UNMATCHED'])
    unmatched_participants = total_participants - matched_participants
    
    if debug_mode:
        print(f"  Total participants: {total_participants}")
        print(f"  Matched participants: {matched_participants}")
        print(f"  Unmatched participants: {unmatched_participants}")
    
    # Step 1: Apply business rules for each participant
    continuing_processed = 0
    non_continuing_processed = 0
    
    for idx, row in df.iterrows():
        # Skip unmatched participants
        circle_id = row['proposed_NEW_circles_id']
        if pd.isna(circle_id) or str(circle_id) == 'UNMATCHED':
            continue
            
        participant_id = row.get('Encoded ID', f'Row_{idx}')
        status = str(row.get('Status', '')).strip()
        
        if status == 'CURRENT-CONTINUING':
            continuing_processed += 1
            
            # Check if they are currently a co-leader
            current_co_leader_col = column_map.get('current_co_leader')
            is_current_co_leader = False
            
            if current_co_leader_col and current_co_leader_col in df.columns:
                co_leader_value = str(row.get(current_co_leader_col, '')).strip().lower()
                is_current_co_leader = co_leader_value == 'yes'
            
            if is_current_co_leader:
                # Check their 2025 response
                response_2025_col = column_map.get('co_leader_response_2025')
                response_2025 = ''
                
                if response_2025_col and response_2025_col in df.columns:
                    response_2025 = str(row.get(response_2025_col, '')).strip().lower()
                
                # If response is "No", set as "No", otherwise set as "Yes"
                if response_2025 == 'no':
                    df.at[idx, 'proposed_NEW_Coleader'] = 'No'
                    if debug_mode:
                        print(f"  CURRENT-CONTINUING Co-Leader {participant_id}: Declined 2025 → No")
                else:
                    df.at[idx, 'proposed_NEW_Coleader'] = 'Yes'
                    if debug_mode:
                        print(f"  CURRENT-CONTINUING Co-Leader {participant_id}: Accepts 2025 → Yes")
            else:
                # Not a current co-leader, but CURRENT-CONTINUING, so they don't become co-leader
                df.at[idx, 'proposed_NEW_Coleader'] = 'No'
                if debug_mode:
                    print(f"  CURRENT-CONTINUING Non-Leader {participant_id}: → No")
        
        else:
            # All non-CURRENT-CONTINUING participants become co-leaders
            non_continuing_processed += 1
            df.at[idx, 'proposed_NEW_Coleader'] = 'Yes'
            if debug_mode and non_continuing_processed <= 5:  # Limit debug output
                print(f"  Non-CONTINUING {participant_id}: Status='{status}' → Yes")
    
    if debug_mode:
        print(f"  Processed {continuing_processed} CURRENT-CONTINUING participants")
        print(f"  Processed {non_continuing_processed} non-CURRENT-CONTINUING participants")
    
    # Step 2: Ensure each circle has at least 2 co-leaders
    matched_df = df[df['proposed_NEW_circles_id'] != 'UNMATCHED']
    circles = matched_df['proposed_NEW_circles_id'].unique()
    circles_needing_adjustment = 0
    
    if debug_mode:
        print(f"\n🔍 CO-LEADER VALIDATION: Checking {len(circles)} circles for minimum co-leaders")
    
    for circle_id in circles:
        circle_members = df[df['proposed_NEW_circles_id'] == circle_id]
        co_leaders = circle_members[circle_members['proposed_NEW_Coleader'] == 'Yes']
        
        if len(co_leaders) < 2:
            circles_needing_adjustment += 1
            # Set all members as co-leaders
            df.loc[df['proposed_NEW_circles_id'] == circle_id, 'proposed_NEW_Coleader'] = 'Yes'
            
            if debug_mode:
                print(f"  Circle {circle_id}: Had {len(co_leaders)} co-leaders, set all {len(circle_members)} as co-leaders")
        elif debug_mode and len(co_leaders) >= 2:
            print(f"  Circle {circle_id}: ✅ Has {len(co_leaders)} co-leaders (sufficient)")
    
    # Final statistics
    final_matched = df[df['proposed_NEW_circles_id'] != 'UNMATCHED']
    final_co_leaders = final_matched[final_matched['proposed_NEW_Coleader'] == 'Yes']
    
    if debug_mode:
        print(f"\n📊 CO-LEADER ASSIGNMENT SUMMARY:")
        print(f"  Circles needing adjustment: {circles_needing_adjustment}")
        print(f"  Total matched participants: {len(final_matched)}")
        print(f"  Total assigned co-leaders: {len(final_co_leaders)}")
        print(f"  Co-leader percentage: {len(final_co_leaders)/len(final_matched)*100:.1f}%")
        
        # Show breakdown by circle
        print(f"\n📋 CO-LEADERS BY CIRCLE:")
        for circle_id in circles[:10]:  # Show first 10 circles
            circle_members = df[df['proposed_NEW_circles_id'] == circle_id]
            circle_co_leaders = circle_members[circle_members['proposed_NEW_Coleader'] == 'Yes']
            print(f"  {circle_id}: {len(circle_co_leaders)}/{len(circle_members)} co-leaders")
    
    return df


def validate_co_leader_assignments(results_df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate that co-leader assignments meet business requirements.
    
    Args:
        results_df: DataFrame with co-leader assignments
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'valid': True,
        'issues': [],
        'statistics': {}
    }
    
    # Check if proposed_NEW_Coleader column exists
    if 'proposed_NEW_Coleader' not in results_df.columns:
        validation_results['valid'] = False
        validation_results['issues'].append("Missing 'proposed_NEW_Coleader' column")
        return validation_results
    
    # Get matched participants only
    matched_df = results_df[results_df['proposed_NEW_circles_id'] != 'UNMATCHED']
    
    if len(matched_df) == 0:
        validation_results['issues'].append("No matched participants found")
        return validation_results
    
    # Check each circle has at least 2 co-leaders
    circles = matched_df['proposed_NEW_circles_id'].unique()
    circles_with_insufficient_leaders = []
    
    for circle_id in circles:
        circle_members = matched_df[matched_df['proposed_NEW_circles_id'] == circle_id]
        co_leaders = circle_members[circle_members['proposed_NEW_Coleader'] == 'Yes']
        
        if len(co_leaders) < 2:
            circles_with_insufficient_leaders.append({
                'circle_id': circle_id,
                'co_leaders': len(co_leaders),
                'total_members': len(circle_members)
            })
    
    if circles_with_insufficient_leaders:
        validation_results['valid'] = False
        validation_results['issues'].append(f"Found {len(circles_with_insufficient_leaders)} circles with < 2 co-leaders")
        validation_results['insufficient_circles'] = circles_with_insufficient_leaders
    
    # Calculate statistics
    total_co_leaders = len(matched_df[matched_df['proposed_NEW_Coleader'] == 'Yes'])
    validation_results['statistics'] = {
        'total_circles': len(circles),
        'total_matched_participants': len(matched_df),
        'total_co_leaders': total_co_leaders,
        'co_leader_percentage': (total_co_leaders / len(matched_df) * 100) if len(matched_df) > 0 else 0,
        'circles_with_insufficient_leaders': len(circles_with_insufficient_leaders)
    }
    
    return validation_results