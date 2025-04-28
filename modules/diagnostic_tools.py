"""
Diagnostic tools for the circle matching algorithm.
This module provides specialized functions to diagnose and troubleshoot matching issues.
"""

import pandas as pd
import streamlit as st
import json
import os

def track_current_continuing_status(region_df, debug_file_path="debug_data/continuing_members_debug.json"):
    """
    Track and analyze all CURRENT-CONTINUING members to diagnose matching issues.
    
    Args:
        region_df: DataFrame with all participants
        debug_file_path: Path to save debug information
        
    Returns:
        dict: Debug information about CURRENT-CONTINUING members
    """
    # Get all CURRENT-CONTINUING members
    continuing_mask = region_df['Status'].isin(['CURRENT-CONTINUING', 'Current-CONTINUING'])
    continuing_df = region_df[continuing_mask]
    
    # Initialize debug container
    debug_info = {
        'total_continuing_members': len(continuing_df),
        'members': []
    }
    
    # Helper to find circle IDs with different methods
    def find_with_method(row, method_name, column_patterns):
        """Find circle ID using specified method and column patterns."""
        for pattern in column_patterns:
            # Try exact match first
            exact_matches = [col for col in row.index if col == pattern]
            if exact_matches:
                col_name = exact_matches[0]
                if not pd.isna(row[col_name]) and row[col_name]:
                    return str(row[col_name]).strip(), col_name
            
            # Try pattern match
            pattern_matches = [col for col in row.index if pattern.lower() in str(col).lower()]
            for col_name in pattern_matches:
                if not pd.isna(row[col_name]) and row[col_name]:
                    return str(row[col_name]).strip(), col_name
        
        return None, None
    
    # Process each CURRENT-CONTINUING member
    for idx, row in continuing_df.iterrows():
        p_id = row['Encoded ID']
        
        # Find circle ID with different methods
        standard_method, standard_col = find_with_method(
            row, 
            "standard", 
            ['Current Circle ID', 'Current_Circle_ID', 'current_circles_id', 'Current Circles ID']
        )
        
        hybrid_method, hybrid_col = find_with_method(
            row,
            "hybrid",
            ['circle current', 'current circle', 'circle id']
        )
        
        # Try the aggressive fallback approach
        aggressive_method = None
        aggressive_col = None
        for col in row.index:
            col_lower = str(col).lower()
            if 'circle' in col_lower:
                if not pd.isna(row[col]) and row[col]:
                    circle_value = str(row[col]).strip()
                    # Check if this looks like a circle ID (contains letters, numbers, and dashes)
                    if '-' in circle_value and any(c.isalpha() for c in circle_value) and any(c.isdigit() for c in circle_value):
                        aggressive_method = circle_value
                        aggressive_col = col
                        break
        
        # Determine best circle ID from all methods
        circle_id = standard_method or hybrid_method or aggressive_method
        circle_id_source = standard_col or hybrid_col or aggressive_col or "None found"
        
        # Add member info to debug container
        member_info = {
            'participant_id': p_id,
            'status': row.get('Status'),
            'region': row.get('Current_Region', row.get('Derived_Region', 'Unknown')),
            'circle_id': circle_id,
            'circle_id_source': circle_id_source,
            'standard_method': {
                'found': standard_method is not None,
                'value': standard_method,
                'column': standard_col
            },
            'hybrid_method': {
                'found': hybrid_method is not None,
                'value': hybrid_method,
                'column': hybrid_col
            },
            'aggressive_method': {
                'found': aggressive_method is not None,
                'value': aggressive_method,
                'column': aggressive_col
            }
        }
        
        debug_info['members'].append(member_info)
    
    # Calculate summary statistics
    debug_info['standard_method_success_rate'] = sum(1 for m in debug_info['members'] if m['standard_method']['found']) / len(debug_info['members']) if debug_info['members'] else 0
    debug_info['hybrid_method_success_rate'] = sum(1 for m in debug_info['members'] if m['hybrid_method']['found']) / len(debug_info['members']) if debug_info['members'] else 0
    debug_info['aggressive_method_success_rate'] = sum(1 for m in debug_info['members'] if m['aggressive_method']['found']) / len(debug_info['members']) if debug_info['members'] else 0
    debug_info['any_method_success_rate'] = sum(1 for m in debug_info['members'] if m['circle_id']) / len(debug_info['members']) if debug_info['members'] else 0
    
    # Create column frequency analysis
    column_analysis = {}
    for m in debug_info['members']:
        if m['circle_id_source'] != "None found":
            if m['circle_id_source'] not in column_analysis:
                column_analysis[m['circle_id_source']] = 0
            column_analysis[m['circle_id_source']] += 1
    
    debug_info['column_frequency'] = column_analysis
    
    # Save debug information to file
    os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)
    with open(debug_file_path, 'w') as f:
        json.dump(debug_info, f, indent=2)
    
    print(f"\n✅ Saved CURRENT-CONTINUING member debug info to {debug_file_path}")
    print(f"📊 Found {debug_info['total_continuing_members']} CURRENT-CONTINUING members")
    print(f"📊 Success rates: Standard={debug_info['standard_method_success_rate']:.2f}, " +
          f"Hybrid={debug_info['hybrid_method_success_rate']:.2f}, " +
          f"Aggressive={debug_info['aggressive_method_success_rate']:.2f}, " +
          f"Any={debug_info['any_method_success_rate']:.2f}")
    
    return debug_info

def track_matching_outcomes(continuing_debug_info, results, unmatched, 
                          debug_file_path="debug_data/continuing_matching_outcomes.json"):
    """
    Track the final matching outcomes for all CURRENT-CONTINUING members.
    
    Args:
        continuing_debug_info: Debug info from track_current_continuing_status
        results: Final matching results
        unmatched: Unmatched participants
        debug_file_path: Path to save debug information
        
    Returns:
        dict: Debug information about matching outcomes
    """
    # Convert results and unmatched to more accessible formats
    results_by_id = {}
    for r in results:
        p_id = r.get('participant_id')
        if p_id:
            results_by_id[p_id] = r
    
    # Initialize outcome container
    outcome_info = {
        'total_continuing_members': continuing_debug_info['total_continuing_members'],
        'matched_members': 0,
        'unmatched_members': 0,
        'matched_to_correct_circle': 0,
        'matched_to_wrong_circle': 0,
        'members': []
    }
    
    # Process each member
    for member in continuing_debug_info['members']:
        p_id = member['participant_id']
        expected_circle = member['circle_id']
        
        # Determine matching outcome
        if p_id in results_by_id:
            # Member was matched
            outcome_info['matched_members'] += 1
            assigned_circle = results_by_id[p_id].get('proposed_NEW_circles_id')
            
            # Compare with expected circle
            matched_correctly = expected_circle and (assigned_circle == expected_circle)
            if matched_correctly:
                outcome_info['matched_to_correct_circle'] += 1
            else:
                outcome_info['matched_to_wrong_circle'] += 1
            
            member_outcome = {
                'participant_id': p_id,
                'expected_circle': expected_circle,
                'assigned_circle': assigned_circle,
                'status': 'MATCHED',
                'matched_correctly': matched_correctly,
                'location_score': results_by_id[p_id].get('location_score'),
                'time_score': results_by_id[p_id].get('time_score'),
                'total_score': results_by_id[p_id].get('total_score')
            }
        else:
            # Member was unmatched
            outcome_info['unmatched_members'] += 1
            unmatched_reason = unmatched.get(p_id, {}).get('unmatched_reason', 'Unknown')
            
            member_outcome = {
                'participant_id': p_id,
                'expected_circle': expected_circle,
                'assigned_circle': 'UNMATCHED',
                'status': 'UNMATCHED',
                'matched_correctly': False,
                'unmatched_reason': unmatched_reason
            }
        
        # Add member outcome to the list
        outcome_info['members'].append({**member, **member_outcome})
    
    # Calculate summary statistics
    if outcome_info['total_continuing_members'] > 0:
        outcome_info['match_rate'] = outcome_info['matched_members'] / outcome_info['total_continuing_members']
        outcome_info['correct_match_rate'] = outcome_info['matched_to_correct_circle'] / outcome_info['total_continuing_members']
    else:
        outcome_info['match_rate'] = 0
        outcome_info['correct_match_rate'] = 0
    
    # Analyze unmatched reasons
    unmatched_reasons = {}
    for member in outcome_info['members']:
        if member.get('status') == 'UNMATCHED':
            reason = member.get('unmatched_reason', 'Unknown')
            if reason not in unmatched_reasons:
                unmatched_reasons[reason] = 0
            unmatched_reasons[reason] += 1
    
    outcome_info['unmatched_reason_frequency'] = unmatched_reasons
    
    # Save outcome information to file
    os.makedirs(os.path.dirname(debug_file_path), exist_ok=True)
    with open(debug_file_path, 'w') as f:
        json.dump(outcome_info, f, indent=2)
    
    print(f"\n✅ Saved CURRENT-CONTINUING matching outcomes to {debug_file_path}")
    print(f"📊 Match rate: {outcome_info['match_rate']:.2f} ({outcome_info['matched_members']} of {outcome_info['total_continuing_members']})")
    print(f"📊 Correct match rate: {outcome_info['correct_match_rate']:.2f} ({outcome_info['matched_to_correct_circle']} of {outcome_info['total_continuing_members']})")
    
    if outcome_info['unmatched_members'] > 0:
        print(f"📊 Top unmatched reasons:")
        for reason, count in sorted(unmatched_reasons.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {reason}: {count} members")
    
    return outcome_info

def add_debug_constraints_log(prob, x, participant_info, circle_info):
    """
    Create detailed logs about which constraints were applied to which participants.
    
    Args:
        prob: PuLP problem object
        x: Dictionary of decision variables
        participant_info: Dictionary mapping participant IDs to their info
        circle_info: Dictionary mapping circle IDs to their info
        
    Returns:
        dict: Debug information about constraints
    """
    # Get all constraints from the problem
    constraints = prob.constraints
    
    # Initialize debug container
    constraint_debug = {
        'total_constraints': len(constraints),
        'constraint_types': {},
        'participant_constraints': {},
        'circle_constraints': {}
    }
    
    # Analyze each constraint
    for name, constraint in constraints.items():
        # Determine constraint type
        constraint_type = 'unknown'
        if 'participant' in name:
            constraint_type = 'participant_assignment'
        elif 'circle' in name:
            constraint_type = 'circle_capacity'
        elif 'host' in name:
            constraint_type = 'host_requirement'
            
        # Count constraint types
        if constraint_type not in constraint_debug['constraint_types']:
            constraint_debug['constraint_types'][constraint_type] = 0
        constraint_debug['constraint_types'][constraint_type] += 1
        
        # Extract participant and circle IDs from the constraint
        participant_ids = []
        circle_ids = []
        
        # Extract variable names from the constraint expression
        for var in constraint.keys():
            var_name = var.name
            if var_name.startswith('x_'):
                # Extract participant and circle IDs from variable name (format: x_participantID_circleID)
                parts = var_name.split('_', 2)
                if len(parts) == 3:
                    participant_id = parts[1]
                    circle_id = parts[2]
                    
                    # Add to lists if not already included
                    if participant_id not in participant_ids:
                        participant_ids.append(participant_id)
                    if circle_id not in circle_ids:
                        circle_ids.append(circle_id)
        
        # Add constraint info to participant constraints
        for p_id in participant_ids:
            if p_id not in constraint_debug['participant_constraints']:
                constraint_debug['participant_constraints'][p_id] = []
            
            constraint_debug['participant_constraints'][p_id].append({
                'constraint_name': name,
                'constraint_type': constraint_type,
                'related_circles': circle_ids
            })
        
        # Add constraint info to circle constraints
        for c_id in circle_ids:
            if c_id not in constraint_debug['circle_constraints']:
                constraint_debug['circle_constraints'][c_id] = []
            
            constraint_debug['circle_constraints'][c_id].append({
                'constraint_name': name,
                'constraint_type': constraint_type,
                'related_participants': participant_ids
            })
    
    # Compute summary statistics
    constraint_debug['average_constraints_per_participant'] = sum(len(v) for v in constraint_debug['participant_constraints'].values()) / len(constraint_debug['participant_constraints']) if constraint_debug['participant_constraints'] else 0
    constraint_debug['average_constraints_per_circle'] = sum(len(v) for v in constraint_debug['circle_constraints'].values()) / len(constraint_debug['circle_constraints']) if constraint_debug['circle_constraints'] else 0
    
    return constraint_debug