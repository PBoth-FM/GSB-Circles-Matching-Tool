"""
GSB Alumni Circle Matching Application
A sophisticated Streamlit application for creating meaningful professional connections
through advanced algorithmic techniques and user-centric design.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from modules.data_loader import load_data, map_column_names, normalize_status_values, validate_data, deduplicate_encoded_ids
from modules.data_processor import process_data, normalize_data
from modules.optimizer import run_matching_algorithm
from utils.circle_id_postprocessor import fix_unknown_circle_ids
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="GSB Alumni Circle Matching",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'config' not in st.session_state:
        st.session_state.config = {
            'min_circle_size': 5,
            'existing_circle_handling': 'optimize',
            'enable_host_requirement': True,
            'debug_mode': False
        }
    
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    if 'results' not in st.session_state:
        st.session_state.results = None
        
    if 'matched_circles' not in st.session_state:
        st.session_state.matched_circles = None
        
    if 'unmatched_participants' not in st.session_state:
        st.session_state.unmatched_participants = None

def clear_session_state():
    """Clear all matching-related session state"""
    keys_to_clear = [
        'results', 'matched_circles', 'unmatched_participants',
        'circle_manager', 'optimization_results', 'circle_eligibility_logs',
        'match_statistics', 'total_diversity_score', 'seattle_debug_logs'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file with participant data"""
    try:
        # Clear previous results
        clear_session_state()
        
        # Load and validate data
        result = load_data(uploaded_file)
        if len(result) == 2:
            df, validation_errors = result
        else:
            df, validation_errors, _ = result  # Handle extra return value
        
        if df is None:
            st.error("Failed to load data from uploaded file.")
            return None
            
        # Track initial count and status breakdown for filtering summary
        initial_count = len(df)
        initial_status_counts = df['Status'].value_counts() if 'Status' in df.columns else None
        
        # Process and normalize data
        processed_data = process_data(df, debug_mode=st.session_state.config.get('debug_mode', False))
        normalized_data = normalize_data(processed_data, debug_mode=st.session_state.config.get('debug_mode', False))
        
        # Track final count and calculate exclusions
        final_count = len(normalized_data)
        excluded_count = initial_count - final_count
        
        # Store filtering info for display
        st.session_state.filtering_info = {
            'initial_count': initial_count,
            'final_count': final_count,
            'excluded_count': excluded_count,
            'initial_status_counts': initial_status_counts
        }
        
        # Store in session state
        st.session_state.processed_data = normalized_data
        
        # Display validation errors if any
        if validation_errors:
            st.warning(f"Found {len(validation_errors)} validation issues:")
            for error in validation_errors[:5]:
                st.write(f"- {error}")
            if len(validation_errors) > 5:
                st.write(f"...and {len(validation_errors) - 5} more issues.")
        
        st.success(f"Data processed successfully! {final_count} participants loaded.")
        
        # Display filtering summary using the actual filter counts from data loading
        if hasattr(st.session_state, 'status_filter_counts') and st.session_state.status_filter_counts:
            filter_counts = st.session_state.status_filter_counts
            total_excluded = filter_counts.get('not_continuing', 0) + filter_counts.get('moving_out', 0)
            
            if total_excluded > 0:
                st.subheader("📊 Data Filtering Summary")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Uploaded Records", initial_count)
                with col2:
                    st.metric("Excluded Records", total_excluded, delta=f"-{total_excluded}")
                with col3:
                    st.metric("Final Participants", final_count)
                
                # Show details of what was excluded
                st.write("**Excluded by status:**")
                if filter_counts.get('not_continuing', 0) > 0:
                    st.write(f"• Not Continuing: {filter_counts['not_continuing']} participants")
                if filter_counts.get('moving_out', 0) > 0:
                    st.write(f"• Moving Out: {filter_counts['moving_out']} participants")
        
        # Display data summary
        st.subheader("Data Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Participants", final_count)
        
        with col2:
            current_continuing = len(normalized_data[normalized_data['Status'] == 'CURRENT-CONTINUING'])
            st.metric("Current Continuing", current_continuing)
        
        with col3:
            new_participants = len(normalized_data[normalized_data['Status'] == 'NEW'])
            st.metric("New Participants", new_participants)
            
        return normalized_data
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def run_optimization():
    """Run the optimization algorithm and store results in session state"""
    if not hasattr(st.session_state, 'processed_data') or st.session_state.processed_data is None:
        st.error("No data available. Please upload and process data first.")
        return
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Starting optimization algorithm...")
        progress_bar.progress(10)
        
        # Run the matching algorithm with better error handling
        status_text.text("Running matching algorithm...")
        progress_bar.progress(30)
        
        results_df, circles_df, unmatched_df = run_matching_algorithm(
            st.session_state.processed_data, 
            st.session_state.config
        )
        
        progress_bar.progress(70)
        status_text.text("Processing results...")
        
        # Store results in session state first
        st.session_state.results = results_df
        st.session_state.matched_circles = circles_df
        st.session_state.unmatched_participants = unmatched_df
        
        progress_bar.progress(90)
        status_text.text("Applying post-processing fixes...")
        
        # Apply post-processing fixes to circle IDs if we have results
        if results_df is not None and not results_df.empty:
            try:
                fixed_results = fix_unknown_circle_ids(results_df)
                st.session_state.results = fixed_results
            except Exception as fix_error:
                st.warning(f"Post-processing warning: {str(fix_error)}")
                # Continue with original results if post-processing fails
        
        progress_bar.progress(100)
        status_text.text("Optimization completed!")
        
        # Clean up progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success("Optimization completed successfully!")
        st.rerun()
        
    except Exception as e:
        # Clean up progress indicators on error
        progress_bar.empty()
        status_text.empty()
        
        st.error(f"Error during optimization: {str(e)}")
        
        # Show detailed error for debugging
        if st.session_state.config.get('debug_mode', False):
            import traceback
            with st.expander("Debug Details"):
                st.code(traceback.format_exc())

def create_circle_composition_table(results_df):
    """Create the Circle Composition table from results CSV data"""
    if results_df is None or results_df.empty:
        st.warning("No results data available for Circle Composition table.")
        return
    
    try:
        # Group by circle ID and calculate metrics
        circle_stats = []
        
        # Get unique circle IDs (using the corrected column)
        circle_id_col = 'proposed_NEW_circles_id'
        if circle_id_col not in results_df.columns:
            st.error(f"Column '{circle_id_col}' not found in results data.")
            return
        
        unique_circles = results_df[circle_id_col].dropna().unique()
        
        for circle_id in unique_circles:
            circle_data = results_df[results_df[circle_id_col] == circle_id]
            
            # Calculate statistics
            member_count = len(circle_data)
            new_members = len(circle_data[circle_data['Status'] == 'NEW'])
            
            # Get region and subregion (use first non-null value)
            region = circle_data['Derived_Region'].dropna().iloc[0] if not circle_data['Derived_Region'].dropna().empty else 'Unknown'
            subregion = circle_data['proposed_NEW_Subregion'].dropna().iloc[0] if not circle_data['proposed_NEW_Subregion'].dropna().empty else 'Unknown'
            
            # Get meeting time
            meeting_time = circle_data['proposed_NEW_DayTime'].dropna().iloc[0] if not circle_data['proposed_NEW_DayTime'].dropna().empty else 'Unknown'
            
            # Calculate max additions (min value of co_leader_max_new_members)
            max_additions_col = 'co_leader_max_new_members'
            if max_additions_col in circle_data.columns:
                max_additions = circle_data[max_additions_col].dropna().min() if not circle_data[max_additions_col].dropna().empty else 0
            else:
                max_additions = 0
            
            # Count host statuses
            host_col = 'host_status_standardized'
            always_hosts = 0
            sometimes_hosts = 0
            if host_col in circle_data.columns:
                always_hosts = len(circle_data[circle_data[host_col] == 'ALWAYS'])
                sometimes_hosts = len(circle_data[circle_data[host_col] == 'SOMETIMES'])
            
            circle_stats.append({
                'Circle ID': circle_id,
                'Region': region,
                'Subregion': subregion,
                'Meeting Time': meeting_time,
                'Member Count': member_count,
                'New Members': new_members,
                'Max Additions': max_additions,
                'Always Hosts': always_hosts,
                'Sometimes Hosts': sometimes_hosts
            })
        
        # Create DataFrame
        circles_df = pd.DataFrame(circle_stats)
        
        if circles_df.empty:
            st.warning("No circle data found.")
            return
        
        # Display the table with column configuration
        st.subheader("Circle Composition")
        
        # Add column width controls
        col1, col2 = st.columns([1, 3])
        with col1:
            auto_resize = st.checkbox("Auto-resize columns", value=True)
        
        if auto_resize:
            # Use default column widths
            st.dataframe(
                circles_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            # Allow manual column width adjustment
            with col2:
                st.info("Manual column width adjustment enabled. Drag column borders to resize.")
            
            st.dataframe(
                circles_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Circle ID': st.column_config.TextColumn(width="medium"),
                    'Region': st.column_config.TextColumn(width="small"),
                    'Subregion': st.column_config.TextColumn(width="medium"),
                    'Meeting Time': st.column_config.TextColumn(width="medium"),
                    'Member Count': st.column_config.NumberColumn(width="small"),
                    'New Members': st.column_config.NumberColumn(width="small"),
                    'Max Additions': st.column_config.NumberColumn(width="small"),
                    'Always Hosts': st.column_config.NumberColumn(width="small"),
                    'Sometimes Hosts': st.column_config.NumberColumn(width="medium")
                }
            )
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Circles", len(circles_df))
        
        with col2:
            avg_size = circles_df['Member Count'].mean()
            st.metric("Average Circle Size", f"{avg_size:.1f}")
        
        with col3:
            total_participants = circles_df['Member Count'].sum()
            st.metric("Total Assigned", total_participants)
        
        with col4:
            new_member_total = circles_df['New Members'].sum()
            st.metric("Total New Members", new_member_total)
        
    except Exception as e:
        st.error(f"Error creating Circle Composition table: {str(e)}")

def data_upload_tab():
    """Display the Data Upload tab content"""
    st.header("Data Upload & Processing")
    
    uploaded_file = st.file_uploader(
        "Upload participant data (CSV)",
        type=['csv'],
        help="Upload a CSV file containing participant information"
    )
    
    if uploaded_file is not None:
        # Process the uploaded file
        processed_data = process_uploaded_file(uploaded_file)
        
        if processed_data is not None:
            st.subheader("Configuration")
            
            # Debug Mode option
            st.session_state.config['debug_mode'] = st.checkbox(
                "Debug Mode", 
                value=st.session_state.config.get('debug_mode', False),
                help="Enable debug output in console"
            )
            
            # Run button
            if st.button("Run Matching Algorithm", key="run_algorithm_button"):
                run_optimization()

def results_tab():
    """Display the Results tab content"""
    st.header("Matching Results")
    
    if not hasattr(st.session_state, 'results') or st.session_state.results is None:
        st.info("No results available. Please upload data and run the matching algorithm first.")
        return
    
    # Display Circle Composition table
    create_circle_composition_table(st.session_state.results)
    
    # Download section
    st.subheader("Download Results")
    
    if hasattr(st.session_state, 'results') and st.session_state.results is not None:
        # Convert to CSV
        csv_buffer = io.StringIO()
        st.session_state.results.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download Results CSV",
            data=csv_data,
            file_name="circle_matching_results.csv",
            mime="text/csv",
            help="Download the complete matching results"
        )

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # App title and description
    st.title("🤝 GSB Alumni Circle Matching")
    st.markdown("Create meaningful professional connections through advanced algorithmic matching")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Data Upload", "📊 Results"])
    
    with tab1:
        data_upload_tab()
    
    with tab2:
        results_tab()

if __name__ == "__main__":
    main()