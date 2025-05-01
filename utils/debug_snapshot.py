import streamlit as st
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import datetime

logger = logging.getLogger(__name__)

def create_debug_snapshot(num_circles: int = 15, specific_circles: Optional[List[str]] = None) -> Dict[str, Any]:
    """Capture current state of circle metadata for debugging
    
    Parameters:
    -----------
    num_circles : int, optional
        Number of circles to sample (default: 15)
    specific_circles : List[str], optional
        List of specific circle IDs to include in the snapshot
        
    Returns:
    --------
    Dict[str, Any]
        Debug snapshot data dictionary
    """
    if "matched_circles" not in st.session_state or "results_df" not in st.session_state:
        logger.warning("No data available for snapshot")
        return {}
    
    snapshot = {}
    snapshot_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "pre_implementation_snapshot"
    }
    
    # Get circle IDs to analyze
    circle_ids = []
    
    # First add any specifically requested circles
    if specific_circles:
        circle_ids.extend([cid for cid in specific_circles 
                         if cid in st.session_state.get("matched_circles", {})])
    
    # Then add additional circles up to num_circles total
    if len(circle_ids) < num_circles:
        all_circle_ids = list(st.session_state.get("matched_circles", {}).keys())
        additional_circles = [cid for cid in all_circle_ids if cid not in circle_ids]
        circle_ids.extend(additional_circles[:num_circles - len(circle_ids)])
    
    # Prioritize any test circles
    test_circles = ["IP-BOS-04", "IP-BOS-05"]
    for test_circle in test_circles:
        if test_circle in st.session_state.get("matched_circles", {}) and test_circle not in circle_ids:
            circle_ids.insert(0, test_circle)  # Add at beginning
            if len(circle_ids) > num_circles:
                circle_ids = circle_ids[:num_circles]  # Keep to requested size
    
    logger.info(f"Creating debug snapshot for {len(circle_ids)} circles")
    
    for circle_id in circle_ids:
        # Get raw members
        raw_members = st.session_state.matched_circles.get(circle_id, [])
        
        # Get UI metadata if available
        ui_metadata = {}
        if "circle_metadata_manager" in st.session_state:
            ui_metadata = st.session_state.circle_metadata_manager.get_circle_data(circle_id)
        
        # Get matched participants for this circle
        member_data = []
        if "results_df" in st.session_state and isinstance(raw_members, list) and raw_members:
            member_data = st.session_state.results_df[
                st.session_state.results_df["Encoded ID"].isin(raw_members)
            ].to_dict('records') if "results_df" in st.session_state else []
        
        # Record the snapshot
        snapshot[circle_id] = {
            "raw_members": raw_members,
            "ui_metadata": ui_metadata.copy() if ui_metadata else {},
            "member_data": member_data
        }
    
    # Add the metadata
    snapshot["_meta"] = snapshot_meta
    
    # Store the snapshot
    st.session_state.pre_refactor_snapshot = snapshot
    logger.info(f"Created debug snapshot of {len(snapshot) - 1} circles")  # -1 for _meta
    
    return snapshot

def compare_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Compare before and after implementation snapshots
    
    Parameters:
    -----------
    before : Dict[str, Any]
        Pre-implementation snapshot
    after : Dict[str, Any]
        Post-implementation snapshot
        
    Returns:
    --------
    Dict[str, Any]
        Comparison results
    """
    comparison = {
        "_meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "before_source": before.get("_meta", {}).get("source", "unknown"),
            "after_source": after.get("_meta", {}).get("source", "unknown")
        },
        "circle_comparisons": {},
        "summary": {}
    }
    
    # Fields to compare
    metadata_fields = [
        "member_count", "new_members", "continuing_members", 
        "always_hosts", "sometimes_hosts", "max_additions"
    ]
    
    # Collect all circle IDs from both snapshots
    all_circle_ids = set(list(before.keys()) + list(after.keys()))
    all_circle_ids.discard("_meta")  # Remove metadata entry
    
    field_change_counts = {field: 0 for field in metadata_fields}
    improved_fields = {field: 0 for field in metadata_fields}
    regression_fields = {field: 0 for field in metadata_fields}
    
    # Compare each circle
    for circle_id in all_circle_ids:
        circle_comparison = {
            "metadata_diff": {},
            "member_diff": {}
        }
        
        # Get before and after metadata
        before_meta = before.get(circle_id, {}).get("ui_metadata", {})
        after_meta = after.get(circle_id, {}).get("ui_metadata", {})
        
        # Compare metadata fields
        for field in metadata_fields:
            before_value = before_meta.get(field)
            after_value = after_meta.get(field)
            
            if before_value != after_value:
                field_change_counts[field] += 1
                circle_comparison["metadata_diff"][field] = {
                    "before": before_value,
                    "after": after_value
                }
                
                # Check if the change is an improvement or regression
                # We define an improvement as a value becoming more accurate or complete
                # This is somewhat subjective and may need refinement
                
                # For host counts, more accurate detection is an improvement
                if field in ["always_hosts", "sometimes_hosts"]:
                    # For now we assume any change is an improvement since we're fixing detection
                    improved_fields[field] += 1
                    
                # For max_additions, matching actual capacity is an improvement
                if field == "max_additions":
                    actual_new = before_meta.get("new_members", 0)
                    before_sufficient = before_value >= actual_new if before_value is not None else False
                    after_sufficient = after_value >= actual_new if after_value is not None else False
                    
                    if not before_sufficient and after_sufficient:
                        improved_fields[field] += 1
                    elif before_sufficient and not after_sufficient:
                        regression_fields[field] += 1
        
        # Compare member lists
        before_members = set(str(m) for m in before.get(circle_id, {}).get("raw_members", []))
        after_members = set(str(m) for m in after.get(circle_id, {}).get("raw_members", []))
        
        if before_members != after_members:
            circle_comparison["member_diff"] = {
                "added": list(after_members - before_members),
                "removed": list(before_members - after_members)
            }
        
        # Only include circles with differences
        if circle_comparison["metadata_diff"] or circle_comparison["member_diff"]:
            comparison["circle_comparisons"][circle_id] = circle_comparison
    
    # Build summary
    comparison["summary"] = {
        "total_circles": len(all_circle_ids),
        "circles_with_differences": len(comparison["circle_comparisons"]),
        "field_changes": field_change_counts,
        "improved_fields": improved_fields,
        "regression_fields": regression_fields
    }
    
    return comparison

def render_snapshot_debug():
    """Render debug UI for snapshot comparison"""
    with st.expander("🔍 Implementation Debug", expanded=False):
        st.markdown("### Debug Snapshots")
        
        cols = st.columns(2)
        with cols[0]:
            if st.button("Create Debug Snapshot"):
                snapshot = create_debug_snapshot()
                st.session_state.debug_snapshot_created = True
                st.session_state.debug_snapshot_timestamp = datetime.datetime.now().isoformat()
                st.experimental_rerun()
        
        with cols[1]:
            if st.button("Compare with Previous") and "pre_refactor_snapshot" in st.session_state:
                # Create a new snapshot for comparison
                new_snapshot = create_debug_snapshot(
                    specific_circles=list(st.session_state.pre_refactor_snapshot.keys())
                )
                new_snapshot["_meta"]["source"] = "post_implementation_snapshot"
                
                # Compare snapshots
                comparison = compare_snapshots(
                    st.session_state.pre_refactor_snapshot,
                    new_snapshot
                )
                
                # Store comparison
                st.session_state.snapshot_comparison = comparison
                st.experimental_rerun()
        
        # Show snapshot info
        if "debug_snapshot_created" in st.session_state and st.session_state.debug_snapshot_created:
            st.success(f"Debug snapshot created at {st.session_state.debug_snapshot_timestamp}")
            st.markdown(f"Snapshot contains {len(st.session_state.pre_refactor_snapshot) - 1} circles")
        
        # Show comparison if available
        if "snapshot_comparison" in st.session_state:
            comp = st.session_state.snapshot_comparison
            summary = comp["summary"]
            
            st.markdown("### Snapshot Comparison Results")
            st.markdown(f"Compared {summary['total_circles']} circles, found differences in {summary['circles_with_differences']}")
            
            # Show field change summary
            st.markdown("#### Field Changes")
            for field, count in summary["field_changes"].items():
                improved = summary["improved_fields"].get(field, 0)
                regressed = summary["regression_fields"].get(field, 0)
                
                status = ""
                if improved > 0 and regressed == 0:
                    status = "🟩 Improved"
                elif regressed > 0 and improved == 0:
                    status = "🔴 Regressed"
                elif improved > 0 and regressed > 0:
                    status = "🟧 Mixed"
                
                st.markdown(f"**{field}**: {count} changes {status}")
            
            # Show per-circle details
            st.markdown("#### Circle Details")
            for circle_id, details in comp["circle_comparisons"].items():
                with st.expander(f"Circle {circle_id}"):
                    st.markdown("**Metadata Changes:**")
                    for field, values in details["metadata_diff"].items():
                        st.markdown(f"{field}: {values['before']} → {values['after']}")
                    
                    if details["member_diff"]:
                        st.markdown("**Member Changes:**")
                        if details["member_diff"].get("added"):
                            st.markdown(f"Added: {', '.join(details['member_diff']['added'])}")
                        if details["member_diff"].get("removed"):
                            st.markdown(f"Removed: {', '.join(details['member_diff']['removed'])}")
