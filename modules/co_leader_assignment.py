"""
Co-Leader Assignment Module

This module implements the business logic for assigning co-leaders to circles
after the optimization algorithm has completed circle assignments.
"""

import pandas as pd
import numpy as np
import re
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
    
    # Map for Co-Leader Response about coming year (pattern-based matching)
    co_leader_response_pattern = r'co-leader\s+response:?\s*cl\s+in\s+(20\d{2})\??'
    matched_column = None
    matched_year = None
    
    for col in df.columns:
        match = re.search(co_leader_response_pattern, col.lower())
        if match:
            year = int(match.group(1))
            # If this is the first match or this year is more recent
            if matched_column is None or year > matched_year:
                matched_column = col
                matched_year = year
    
    if matched_column:
        column_map['co_leader_response_coming_year'] = matched_column
    
    # Map for Non-CLs Volunteering to Co-Lead
    volunteering_variations = [
        '(Non CLs) Volunteering to Co-Lead?', 'Volunteering to Co-Lead?',
        'Volunteering to Co-Lead', 'Non CLs Volunteering to Co-Lead'
    ]
    for col in volunteering_variations:
        if col in df.columns:
            column_map['volunteering_to_co_lead'] = col
            break
    
    # Map for Sole Leader Willingness
    sole_leader_column = 'If only CL volunteer, willing to be sole Leader? (Y/N)'
    if sole_leader_column in df.columns:
        column_map['willing_sole_leader'] = sole_leader_column
    
    return column_map


def assign_co_leaders(results_df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
    """
    Assign co-leaders to each circle based on the specified business rules.
    
    Business Rules:
    1. For CURRENT-CONTINUING participants who are current co-leaders:
       - If "Co-Leader Response: CL in YYYY?" is "No" → proposed_NEW_Coleader = "No"
       - Otherwise → proposed_NEW_Coleader = "Yes"
    
    2. For non-CURRENT-CONTINUING participants:
       - If "(Non CLs) Volunteering to Co-Lead?" is "Yes" → proposed_NEW_Coleader = "Yes"
       - Otherwise → proposed_NEW_Coleader = "No"
    
    3. Enhanced minimum co-leader rules for circles with < 2 co-leaders:
       - If exactly 1 person eligible AND willing to be sole leader → Make only them co-leader
       - If multiple people eligible → Make all eligible people co-leaders
       - Otherwise → Safety net: make ALL participants in circle co-leaders
    
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
        if 'co_leader_response_coming_year' in column_map:
            detected_col = column_map['co_leader_response_coming_year']
            # Extract year for logging
            year_match = re.search(r'(20\d{2})', detected_col)
            detected_year = year_match.group(1) if year_match else 'unknown'
            print(f"  Detected co-leader response column: '{detected_col}' (year: {detected_year})")
    
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
                # Check their coming year response
                response_coming_year_col = column_map.get('co_leader_response_coming_year')
                response_coming_year = ''
                
                if response_coming_year_col and response_coming_year_col in df.columns:
                    response_coming_year = str(row.get(response_coming_year_col, '')).strip().lower()
                
                # If response is "No", set as "No", otherwise set as "Yes"
                if response_coming_year == 'no':
                    df.at[idx, 'proposed_NEW_Coleader'] = 'No'
                    if debug_mode:
                        print(f"  CURRENT-CONTINUING Co-Leader {participant_id}: Declined coming year → No")
                else:
                    df.at[idx, 'proposed_NEW_Coleader'] = 'Yes'
                    if debug_mode:
                        print(f"  CURRENT-CONTINUING Co-Leader {participant_id}: Accepts coming year → Yes")
            else:
                # Not a current co-leader, but CURRENT-CONTINUING, so they don't become co-leader
                df.at[idx, 'proposed_NEW_Coleader'] = 'No'
                if debug_mode:
                    print(f"  CURRENT-CONTINUING Non-Leader {participant_id}: → No")
        
        else:
            # For non-CURRENT-CONTINUING participants, check if they volunteered to co-lead
            non_continuing_processed += 1
            
            # Check volunteering status
            volunteering_col = column_map.get('volunteering_to_co_lead')
            volunteered = False
            
            if volunteering_col and volunteering_col in df.columns:
                volunteer_value = str(row.get(volunteering_col, '')).strip().lower()
                volunteered = volunteer_value == 'yes'
            
            # Apply business rule: Check volunteering response
            if volunteered:
                df.at[idx, 'proposed_NEW_Coleader'] = 'Yes'
            else:
                df.at[idx, 'proposed_NEW_Coleader'] = 'No'
            
            if debug_mode and non_continuing_processed <= 5:  # Limit debug output
                volunteer_status = "volunteered" if volunteered else "did not volunteer"
                result = "Yes" if volunteered else "No"
                print(f"  Non-CONTINUING {participant_id}: Status='{status}', {volunteer_status} → {result}")
    
    if debug_mode:
        print(f"  Processed {continuing_processed} CURRENT-CONTINUING participants")
        print(f"  Processed {non_continuing_processed} non-CURRENT-CONTINUING participants")
    
    # Step 2: Enhanced co-leader assignment with sole leader logic
    matched_df = df[df['proposed_NEW_circles_id'] != 'UNMATCHED']
    circles = matched_df['proposed_NEW_circles_id'].unique()
    circles_needing_adjustment = 0
    sole_leader_scenarios = 0
    
    if debug_mode:
        print(f"\n🔍 CO-LEADER VALIDATION: Checking {len(circles)} circles for minimum co-leaders")
    
    sole_leader_col = column_map.get('willing_sole_leader')
    
    for circle_id in circles:
        circle_members = df[df['proposed_NEW_circles_id'] == circle_id]
        co_leaders = circle_members[circle_members['proposed_NEW_Coleader'] == 'Yes']
        
        if len(co_leaders) < 2:
            circles_needing_adjustment += 1
            
            # Find all eligible people (those who would be co-leaders under Step 1 rules)
            # This is the same as co_leaders since we're before any adjustments
            eligible_members = co_leaders
            
            if debug_mode:
                print(f"  Circle {circle_id}: Has {len(co_leaders)} co-leaders, {len(eligible_members)} eligible")
            
            if len(eligible_members) == 1:
                # Exactly one eligible person - check sole leader willingness
                eligible_person = eligible_members.iloc[0]
                willing_sole_leader = False
                
                if sole_leader_col and sole_leader_col in df.columns:
                    sole_leader_response = str(eligible_person.get(sole_leader_col, '')).strip().lower()
                    willing_sole_leader = sole_leader_response == 'yes' or sole_leader_response == 'y'
                
                if willing_sole_leader:
                    # Make only this person co-leader, others stay as "No"
                    sole_leader_scenarios += 1
                    if debug_mode:
                        participant_id = eligible_person.get('Encoded ID', 'Unknown')
                        print(f"  Circle {circle_id}: 👑 SOLE LEADER - {participant_id} willing to lead alone")
                else:
                    # Apply safety net: make all members co-leaders
                    df.loc[df['proposed_NEW_circles_id'] == circle_id, 'proposed_NEW_Coleader'] = 'Yes'
                    if debug_mode:
                        participant_id = eligible_person.get('Encoded ID', 'Unknown')
                        print(f"  Circle {circle_id}: {participant_id} not willing to be sole leader → all {len(circle_members)} members as co-leaders")
                        
            elif len(eligible_members) > 1:
                # Multiple eligible people - make all eligible co-leaders (they already are)
                if debug_mode:
                    print(f"  Circle {circle_id}: Multiple eligible ({len(eligible_members)}) → all eligible remain co-leaders")
                    
            else:
                # Zero eligible people - apply safety net: make all members co-leaders
                df.loc[df['proposed_NEW_circles_id'] == circle_id, 'proposed_NEW_Coleader'] = 'Yes'
                if debug_mode:
                    print(f"  Circle {circle_id}: No eligible members → all {len(circle_members)} members as co-leaders")
                    
        elif debug_mode and len(co_leaders) >= 2:
            print(f"  Circle {circle_id}: ✅ Has {len(co_leaders)} co-leaders (sufficient)")
    
    # Final statistics
    final_matched = df[df['proposed_NEW_circles_id'] != 'UNMATCHED']
    final_co_leaders = final_matched[final_matched['proposed_NEW_Coleader'] == 'Yes']
    
    if debug_mode:
        print(f"\n📊 CO-LEADER ASSIGNMENT SUMMARY:")
        print(f"  Circles needing adjustment: {circles_needing_adjustment}")
        print(f"  Sole leader scenarios: {sole_leader_scenarios}")
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
    
    # Check each circle has adequate leadership (2+ co-leaders OR 1 willing sole leader)
    circles = matched_df['proposed_NEW_circles_id'].unique()
    circles_with_insufficient_leaders = []
    
    # Find sole leader column for validation
    column_map = find_co_leader_columns(results_df)
    sole_leader_col = column_map.get('willing_sole_leader')
    
    for circle_id in circles:
        circle_members = matched_df[matched_df['proposed_NEW_circles_id'] == circle_id]
        co_leaders = circle_members[circle_members['proposed_NEW_Coleader'] == 'Yes']
        
        if len(co_leaders) < 2:
            # Check if this is a valid sole leader scenario
            is_valid_sole_leader = False
            
            if len(co_leaders) == 1 and sole_leader_col and sole_leader_col in results_df.columns:
                sole_leader = co_leaders.iloc[0]
                sole_leader_response = str(sole_leader.get(sole_leader_col, '')).strip().lower()
                is_valid_sole_leader = sole_leader_response == 'yes' or sole_leader_response == 'y'
            
            if not is_valid_sole_leader:
                circles_with_insufficient_leaders.append({
                    'circle_id': circle_id,
                    'co_leaders': len(co_leaders),
                    'total_members': len(circle_members),
                    'is_sole_leader_scenario': len(co_leaders) == 1,
                    'sole_leader_willing': is_valid_sole_leader if len(co_leaders) == 1 else None
                })
    
    if circles_with_insufficient_leaders:
        validation_results['valid'] = False
        validation_results['issues'].append(f"Found {len(circles_with_insufficient_leaders)} circles with insufficient leadership (neither 2+ co-leaders nor willing sole leader)")
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