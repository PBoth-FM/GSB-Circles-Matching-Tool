import streamlit as st
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

# Configure logging
logger = logging.getLogger('debug_snapshot')

def capture_snapshot(label: str, data: Dict[str, Any]) -> None:
    """
    Capture a debug snapshot of current application state
    
    Args:
        label: Unique identifier for this snapshot
        data: Dictionary of data to capture (must be serializable)
    """
    # Initialize snapshots in session state if needed
    if 'debug_snapshots' not in st.session_state:
        st.session_state.debug_snapshots = {}
    
    # Add timestamp
    snapshot = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data': data
    }
    
    # Store snapshot
    st.session_state.debug_snapshots[label] = snapshot
    logger.info(f"Captured debug snapshot: {label}")

def get_snapshot(label: str) -> Optional[Dict[str, Any]]:
    """
    Get a captured debug snapshot
    
    Args:
        label: Label of the snapshot to retrieve
        
    Returns:
        Snapshot data or None if not found
    """
    if 'debug_snapshots' not in st.session_state or label not in st.session_state.debug_snapshots:
        return None
    
    return st.session_state.debug_snapshots[label]

def get_all_snapshots() -> Dict[str, Dict[str, Any]]:
    """
    Get all captured debug snapshots
    
    Returns:
        Dictionary of all snapshots by label
    """
    if 'debug_snapshots' not in st.session_state:
        return {}
    
    return st.session_state.debug_snapshots

def clear_snapshots() -> None:
    """
    Clear all debug snapshots
    """
    if 'debug_snapshots' in st.session_state:
        st.session_state.debug_snapshots = {}
        logger.info("Cleared all debug snapshots")

def compare_snapshots(before_label: str, after_label: str) -> Dict[str, Any]:
    """
    Compare two snapshots to identify differences
    
    Args:
        before_label: Label of the "before" snapshot
        after_label: Label of the "after" snapshot
        
    Returns:
        Dictionary of differences between snapshots
    """
    before = get_snapshot(before_label)
    after = get_snapshot(after_label)
    
    if not before or not after:
        return {
            'error': 'One or both snapshots not found',
            'before_exists': before is not None,
            'after_exists': after is not None
        }
    
    # Compare timestamps
    result = {
        'before_timestamp': before['timestamp'],
        'after_timestamp': after['timestamp'],
        'differences': {}
    }
    
    # Compare data keys
    all_keys = set(before['data'].keys()) | set(after['data'].keys())
    
    for key in all_keys:
        # Track differences for this key
        if key not in before['data']:
            result['differences'][key] = {
                'type': 'added',
                'after_value': after['data'][key]
            }
        elif key not in after['data']:
            result['differences'][key] = {
                'type': 'removed',
                'before_value': before['data'][key]
            }
        elif before['data'][key] != after['data'][key]:
            result['differences'][key] = {
                'type': 'changed',
                'before_value': before['data'][key],
                'after_value': after['data'][key]
            }
    
    return result

def render_snapshots() -> None:
    """
    Render debug snapshots UI
    """
    st.subheader("Debug Snapshots")
    
    snapshots = get_all_snapshots()
    
    if not snapshots:
        st.info("No debug snapshots captured yet")
        return
    
    # Display available snapshots
    st.markdown("### Available Snapshots")
    
    for label, snapshot in snapshots.items():
        st.markdown(f"**{label}** - {snapshot['timestamp']}")
    
    # Comparison UI
    st.markdown("### Compare Snapshots")
    
    snapshot_labels = list(snapshots.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        before_label = st.selectbox("Before snapshot", snapshot_labels, key="before_snapshot")
    
    with col2:
        # Default the after snapshot to something different than before, if possible
        default_after = 1 if len(snapshot_labels) > 1 else 0
        after_label = st.selectbox("After snapshot", snapshot_labels, index=default_after, key="after_snapshot")
    
    if st.button("Compare Snapshots"):
        if before_label == after_label:
            st.warning("Please select different snapshots to compare")
        else:
            comparison = compare_snapshots(before_label, after_label)
            
            st.markdown(f"Comparing **{before_label}** ({comparison['before_timestamp']}) with **{after_label}** ({comparison['after_timestamp']})")
            
            if 'error' in comparison:
                st.error(comparison['error'])
            elif not comparison['differences']:
                st.success("No differences found between snapshots")
            else:
                st.markdown(f"Found {len(comparison['differences'])} differences:")
                
                for key, diff in comparison['differences'].items():
                    st.markdown(f"**{key}** - {diff['type']}")
                    
                    if diff['type'] == 'added':
                        st.json(diff['after_value'])
                    elif diff['type'] == 'removed':
                        st.json(diff['before_value'])
                    else:  # changed
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("Before:")
                            st.json(diff['before_value'])
                        with col2:
                            st.markdown("After:")
                            st.json(diff['after_value'])
    
    # Clear snapshots button
    if st.button("Clear All Snapshots"):
        clear_snapshots()
        st.experimental_rerun()

def capture_circle_metadata(label: str, manager) -> None:
    """
    Convenience function to capture a snapshot of circle metadata
    
    Args:
        label: Unique identifier for this snapshot
        manager: CircleMetadataManager instance
    """
    circles = manager.get_all_circles()
    
    # Clean up data for storage (convert numerical values to ensure serializability)
    for circle in circles:
        for key, value in circle.items():
            if isinstance(value, (pd.Series, pd.DataFrame)):
                circle[key] = value.to_dict()
    
    # Capture statistics
    all_host_counts = {
        'always_hosts': [c.get('always_hosts', 0) for c in circles],
        'sometimes_hosts': [c.get('sometimes_hosts', 0) for c in circles]
    }
    
    snapshot_data = {
        'circle_count': len(circles),
        'circle_summary': {
            'always_hosts_sum': sum(all_host_counts['always_hosts']),
            'sometimes_hosts_sum': sum(all_host_counts['sometimes_hosts']),
            'always_hosts_avg': sum(all_host_counts['always_hosts']) / len(circles) if circles else 0,
            'sometimes_hosts_avg': sum(all_host_counts['sometimes_hosts']) / len(circles) if circles else 0,
        },
        'circle_metadata': circles,
    }
    
    capture_snapshot(label, snapshot_data)
    logger.info(f"Captured circle metadata snapshot: {label} with {len(circles)} circles")
