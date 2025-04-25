import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from modules.data_loader import load_data, validate_data
from modules.data_processor import process_data, normalize_data
from modules.optimizer import run_matching_algorithm
from modules.ui_components import (
    render_match_tab, 
    render_details_tab, 
    render_debug_tab,
    render_demographics_tab,
    render_results_overview,
    render_circle_table,
    render_unmatched_table
)
from utils.helpers import generate_download_link

# Configure Streamlit page
st.set_page_config(
    page_title="CirclesTool2",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'matched_circles' not in st.session_state:
    st.session_state.matched_circles = None
if 'unmatched_participants' not in st.session_state:
    st.session_state.unmatched_participants = None
if 'validation_errors' not in st.session_state:
    st.session_state.validation_errors = None
if 'deduplication_messages' not in st.session_state:
    st.session_state.deduplication_messages = []
if 'exec_time' not in st.session_state:
    st.session_state.exec_time = None
if 'circle_eligibility_logs' not in st.session_state:
    st.session_state.circle_eligibility_logs = {}
if 'config' not in st.session_state:
    st.session_state.config = {
        'debug_mode': False,
        'min_circle_size': 5,
        'existing_circle_handling': 'preserve',
        'optimization_weight_location': 3,
        'optimization_weight_time': 2,
        'enable_host_requirement': True
    }

def main():
    st.title("CirclesTool2")
    st.write("GSB Alumni Circle Matching Tool")
    
    # Create tabs for navigation, moved Demographics after Match per user request
    tab1, tab2, tab3, tab4 = st.tabs(["Match", "Demographics", "Details", "Debug"])
    
    with tab1:
        # Use our custom match tab function instead of the imported one
        match_tab_callback()
        
    with tab2:
        render_demographics_tab()
    
    with tab3:
        render_details_tab()
            
    with tab4:
        render_debug_tab()

def run_optimization():
    """Run the optimization algorithm and store results in session state"""
    # Reset debug logs at the start of each optimization run
    # We're no longer importing the global circle_eligibility_logs variable
    from modules.optimizer_new import debug_eligibility_logs
    from modules.optimizer_new import update_session_state_eligibility_logs
    
    # We now manage logs through session state directly instead of a global variable
    # Initialize the logs dictionary in session state if needed
    import streamlit as st
    if 'circle_eligibility_logs' not in st.session_state:
        st.session_state.circle_eligibility_logs = {}
    
    # Clear the existing session state logs
    st.session_state.circle_eligibility_logs.clear()
    print(f"🧹 Cleared previous logs from session state")
    
    # Log the reset for debugging
    debug_eligibility_logs("Cleared circle eligibility logs before optimization run")
        
    # Log the reset for debugging
    print("🔄 CRITICAL DEBUG: Reset circle eligibility logs before optimization run")
    try:
        with st.spinner("Running matching algorithm..."):
            start_time = time.time()
            
            # Run the matching algorithm with enhanced return values for debugging
            results, matched_circles, unmatched_participants = run_matching_algorithm(
                st.session_state.processed_data,
                st.session_state.config
            )
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.matched_circles = matched_circles
            st.session_state.unmatched_participants = unmatched_participants
            st.session_state.exec_time = time.time() - start_time
            
            # DEBUGGING: Check if we actually have eligibility logs in session state
            print(f"After optimization, circle_eligibility_logs contains {len(st.session_state.circle_eligibility_logs)} entries")
            if len(st.session_state.circle_eligibility_logs) == 0:
                print("WARNING: No circle eligibility logs found in session state!")
                
                # Try loading from file
                try:
                    from modules.optimizer_new import load_circle_eligibility_logs_from_file
                    file_logs = load_circle_eligibility_logs_from_file()
                    if file_logs and len(file_logs) > 0:
                        print(f"✅ Loaded {len(file_logs)} logs from file")
                        st.session_state.circle_eligibility_logs = file_logs
                    else:
                        print("ℹ️ No logs found in file, generating test data")
                        # Create and save test data for debugging
                        from modules.optimizer_new import save_circle_eligibility_logs_to_file
                        
                        # Create test circle eligibility logs
                        test_logs = {
                            'IP-TEST-01': {
                                'circle_id': 'IP-TEST-01',
                                'region': 'Test Region',
                                'subregion': 'Test Subregion',
                                'is_eligible': True,
                                'current_members': 7,
                                'max_additions': 3,
                                'is_small_circle': False,
                                'is_test_circle': True,
                                'has_none_preference': False,
                                'preference_overridden': False,
                                'meeting_time': 'Monday (Evening)',
                                'reason': 'Has capacity'
                            },
                            'IP-TEST-02': {
                                'circle_id': 'IP-TEST-02',
                                'region': 'Test Region',
                                'subregion': 'Test Subregion',
                                'is_eligible': False,
                                'current_members': 10,
                                'max_additions': 0,
                                'is_small_circle': False,
                                'is_test_circle': True,
                                'has_none_preference': True,
                                'preference_overridden': False,
                                'reason': 'Circle is at maximum capacity (10 members)',
                                'meeting_time': 'Wednesday (Evening)'
                            },
                            'IP-TEST-03': {
                                'circle_id': 'IP-TEST-03',
                                'region': 'Test Region',
                                'subregion': 'Test Subregion',
                                'is_eligible': True,
                                'current_members': 4,
                                'max_additions': 6,
                                'is_small_circle': True,
                                'is_test_circle': True,
                                'has_none_preference': True,
                                'preference_overridden': True,
                                'override_reason': 'Small circle override applied',
                                'meeting_time': 'Friday (Evening)',
                                'reason': 'Small circle needs to reach viable size'
                            }
                        }
                        
                        # Save to file for testing
                        saved = save_circle_eligibility_logs_to_file(test_logs, "Test Region")
                        if saved:
                            print(f"✅ Successfully saved {len(test_logs)} test logs to file")
                            # Update session state with the test logs
                            st.session_state.circle_eligibility_logs = test_logs
                        else:
                            print("❌ Failed to save test logs to file")
                except Exception as e:
                    print(f"❌ Error during file operations: {str(e)}")
                    print("Please check if optimizer_new.py's update_session_state_eligibility_logs function was called properly")
            
            st.success(f"Matching completed in {st.session_state.exec_time:.2f} seconds!")
            st.session_state.active_tab = "Results"
            
    except Exception as e:
        st.error(f"Error during optimization: {str(e)}")
        if st.session_state.config['debug_mode']:
            st.exception(e)

def process_uploaded_file(uploaded_file):
    """Process the uploaded CSV file with participant data"""
    try:
        # Force clear any cached values
        st.cache_data.clear()
        with st.spinner("Processing data..."):
            # Load and validate data
            df, validation_errors, deduplication_messages = load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.validation_errors = validation_errors
            st.session_state.deduplication_messages = deduplication_messages
            
            # Count participants by status for debugging
            if 'Status' in df.columns:
                status_counts = df['Status'].value_counts().to_dict()
                st.session_state.status_counts = status_counts
            
            # Display validation errors if any
            if len(validation_errors) > 0:
                st.warning(f"Found {len(validation_errors)} validation issues:")
                for error in validation_errors[:5]:  # Show first 5 errors
                    st.write(f"- {error}")
                if len(validation_errors) > 5:
                    st.write(f"...and {len(validation_errors) - 5} more issues.")
            
            # Display just the count of fixed duplicate Encoded IDs
            if len(deduplication_messages) > 0:
                st.warning(f"Found and fixed {len(deduplication_messages)} duplicate Encoded IDs")
            
            # Process and normalize data - pass debug_mode from session state if available
            debug_mode = st.session_state.get('debug_mode', False)
            
            processed_data = process_data(df, debug_mode=debug_mode)
            
            # Check for "Moving out" status records that will be excluded
            moving_out_count = 0
            if 'Status' in df.columns:
                for status in df['Status']:
                    if isinstance(status, str) and "MOVING OUT" in status.upper():
                        moving_out_count += 1
                
                if moving_out_count > 0:
                    st.info(f"{moving_out_count} records with 'Moving Out' status will be excluded from matching")
            
            normalized_data = normalize_data(processed_data, debug_mode=debug_mode)
            st.session_state.processed_data = normalized_data
            
            st.success(f"Data processed successfully! {len(normalized_data)} participants loaded.")
            
            # Display data summary
            st.subheader("Data Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Participants", len(normalized_data))
            
            with col2:
                current_continuing = len(normalized_data[normalized_data['Status'] == 'CURRENT-CONTINUING'])
                st.metric("Current Continuing", current_continuing)
            
            with col3:
                new_participants = len(normalized_data[normalized_data['Status'] == 'NEW'])
                st.metric("New Participants", new_participants)
            
            st.subheader("Configuration")
            # Keep only Debug Mode and set other options to fixed values
            st.session_state.config['min_circle_size'] = 5  # Fixed value
            st.session_state.config['existing_circle_handling'] = 'preserve'  # Fixed value
            st.session_state.config['enable_host_requirement'] = True  # Fixed value
            
            # Only show Debug Mode as a configurable option
            st.session_state.config['debug_mode'] = st.checkbox(
                "Debug Mode", 
                value=st.session_state.config['debug_mode'],
                help="Enable to see detailed logs and diagnostic information"
            )
            
            # Add run button once data is loaded
            if st.button("Run Matching Algorithm", key="match_run_algorithm_button"):
                run_optimization()
                
            # Display results if available
            if st.session_state.results is not None:
                # We don't need to render the overview in the match tab
                # render_results_overview() - removed to avoid duplicate charts
                
                # Display stats about the matching
                if 'results' in st.session_state and st.session_state.results is not None:
                    results_df = st.session_state.results
                    total_participants = len(results_df)
                    matched_count = len(results_df[results_df['proposed_NEW_circles_id'] != 'UNMATCHED']) if 'proposed_NEW_circles_id' in results_df.columns else 0
                    unmatched_count = len(results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED']) if 'proposed_NEW_circles_id' in results_df.columns else 0
                    
                    # Create columns for the metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
                            st.metric("Circles Created", len(st.session_state.matched_circles))
                    
                    with col2:
                        st.metric("Participants Matched", matched_count)
                    
                    with col3:
                        # Directly calculate all diversity scores instead of using the central function
                        # Import all necessary functions
                        from modules.ui_components import calculate_vintage_diversity_score
                        from modules.ui_components import calculate_employment_diversity_score
                        from modules.ui_components import calculate_industry_diversity_score
                        from modules.ui_components import calculate_racial_identity_diversity_score
                        from modules.ui_components import calculate_children_diversity_score
                        
                        # Get the matched circles and results data
                        matched_circles_df = st.session_state.matched_circles
                        results_df = st.session_state.results
                        
                        # Calculate each diversity score individually
                        vintage_score = calculate_vintage_diversity_score(matched_circles_df, results_df)
                        employment_score = calculate_employment_diversity_score(matched_circles_df, results_df)
                        industry_score = calculate_industry_diversity_score(matched_circles_df, results_df)
                        ri_score = calculate_racial_identity_diversity_score(matched_circles_df, results_df)
                        children_score = calculate_children_diversity_score(matched_circles_df, results_df)
                        
                        # Calculate total score as the sum
                        total_diversity_score = vintage_score + employment_score + industry_score + ri_score + children_score
                        
                        # Store the scores in session state for use in other parts of the app
                        st.session_state.vintage_diversity_score = vintage_score
                        st.session_state.employment_diversity_score = employment_score
                        st.session_state.industry_diversity_score = industry_score
                        st.session_state.racial_identity_diversity_score = ri_score
                        st.session_state.children_diversity_score = children_score
                        
                        # Display Diversity Score metric
                        st.metric("Diversity Score", total_diversity_score)
                        
                        # Log for debugging
                        print(f"DEBUG - Match page diversity scores: Vintage({vintage_score}) + Employment({employment_score}) + " +
                              f"Industry({industry_score}) + RI({ri_score}) + Children({children_score}) = Total({total_diversity_score})")
                        
                    with col4:
                        if total_participants > 0:
                            match_rate = (matched_count / total_participants) * 100
                            st.metric("Match Success Rate", f"{match_rate:.1f}%")
                
                st.subheader("Circle Composition")
                
                # Display circle table with specified columns
                if ('matched_circles' in st.session_state and 
                    st.session_state.matched_circles is not None and 
                    not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty)):
                    
                    # Get the data
                    circles_df = st.session_state.matched_circles.copy()
                    
                    # Extract key columns as specified
                    display_cols = ['circle_id', 'region', 'subregion', 'meeting_time', 'member_count', 'new_members', 'max_additions', 'always_hosts', 'sometimes_hosts']
                    
                    # Filter to only include columns that exist in the dataframe
                    existing_cols = [col for col in display_cols if col in circles_df.columns]
                    
                    if existing_cols:
                        display_df = circles_df[existing_cols].copy()
                        
                        # Rename columns for display
                        display_df.columns = [col.replace('_', ' ').title() for col in existing_cols]
                        
                        # Sort by circle ID if available
                        if 'Circle Id' in display_df.columns:
                            display_df = display_df.sort_values('Circle Id')
                        
                        # Show the table
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.warning("Circle data doesn't contain the expected columns.")
                else:
                    st.warning("No matching results available. Please run the matching algorithm first.")
                
                # Display unmatched participants
                st.subheader("Unmatched Participants")
                
                if 'results' in st.session_state and st.session_state.results is not None:
                    results_df = st.session_state.results
                    
                    if 'proposed_NEW_circles_id' in results_df.columns:
                        unmatched_df = results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED'].reset_index(drop=True)
                        
                        if len(unmatched_df) == 0:
                            st.success("All participants were successfully matched!")
                        else:
                            st.write(f"Total unmatched: {len(unmatched_df)} participants")
                            
                            # Create a display DataFrame with just the columns we need
                            # We'll get the specific columns from the Results CSV by position
                            
                            # Get all column names from the DataFrame for debugging
                            all_columns = list(unmatched_df.columns)
                            
                            # Define the display columns based on their position in the CSV
                            display_cols = []
                            
                            # Encoded ID (column 3 in CSV) - Identify by name
                            encoded_id_col = None
                            if 'Encoded ID' in all_columns:
                                encoded_id_col = 'Encoded ID'
                                display_cols.append(encoded_id_col)
                            
                            # Try to find Derived_Region (column 32 in CSV file)
                            # First approach: Try to find by column name
                            region_col = None
                            region_candidates = ['Derived_Region', 'Derived Region', 'Region Requested', 'Region']
                            for col in region_candidates:
                                if col in all_columns:
                                    region_col = col
                                    display_cols.append(region_col)
                                    break
                                    
                            # Alternative approach: Enable debug to see all column names
                            # Uncomment to see columns when debugging
                            # st.write("All columns in dataframe:", all_columns)
                            
                            # Reason Unmatched (column 5 in CSV)
                            reason_col = None
                            if 'unmatched_reason' in all_columns:
                                reason_col = 'unmatched_reason'
                                display_cols.append(reason_col)
                            
                            # Find choice columns - these need to be identified by position or pattern
                            
                            # 1st, 2nd, 3rd Choice Location columns
                            location_cols = []
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "location_choice_1", "meeting_location_1", "location_pref_1", 
                                    "1st_choice_location", "first_choice_location"
                                ]):
                                    location_cols.append(col)
                                    break
                            
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "location_choice_2", "meeting_location_2", "location_pref_2", 
                                    "2nd_choice_location", "second_choice_location"
                                ]):
                                    location_cols.append(col)
                                    break
                                    
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "location_choice_3", "meeting_location_3", "location_pref_3", 
                                    "3rd_choice_location", "third_choice_location"
                                ]):
                                    location_cols.append(col)
                                    break
                            
                            # 1st, 2nd, 3rd Choice Time columns
                            time_cols = []
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "time_choice_1", "meeting_time_1", "time_pref_1", 
                                    "1st_choice_time", "first_choice_time"
                                ]):
                                    time_cols.append(col)
                                    break
                            
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "time_choice_2", "meeting_time_2", "time_pref_2", 
                                    "2nd_choice_time", "second_choice_time"
                                ]):
                                    time_cols.append(col)
                                    break
                                    
                            for col in all_columns:
                                if any(pattern in col.lower() for pattern in [
                                    "time_choice_3", "meeting_time_3", "time_pref_3", 
                                    "3rd_choice_time", "third_choice_time"
                                ]):
                                    time_cols.append(col)
                                    break
                            
                            # As a fallback, try to find columns by pattern if positional search failed
                            if not location_cols or not time_cols:
                                # Find all location and time columns
                                pattern_location_cols = []
                                pattern_time_cols = []
                                
                                for col in all_columns:
                                    if any(pattern in col.lower() for pattern in ['location_choice', 'meeting_location', 'location_pref']):
                                        pattern_location_cols.append(col)
                                    if any(pattern in col.lower() for pattern in ['time_choice', 'meeting_time', 'time_pref']):
                                        pattern_time_cols.append(col)
                                
                                # Sort them to ensure they're in order (1st, 2nd, 3rd)
                                pattern_location_cols.sort()
                                pattern_time_cols.sort()
                                
                                # Use up to 3 of each
                                location_cols = pattern_location_cols[:3]
                                time_cols = pattern_time_cols[:3]
                            
                            # Add the location and time columns to display_cols, interleaving them
                            for i in range(min(3, max(len(location_cols), len(time_cols)))):
                                if i < len(location_cols):
                                    display_cols.append(location_cols[i])
                                if i < len(time_cols):
                                    display_cols.append(time_cols[i])
                            
                            # Remove any columns that don't exist in the dataframe
                            display_cols = [col for col in display_cols if col in all_columns]
                            
                            # Debug info
                            # st.write(f"All columns: {all_columns}")
                            # st.write(f"Display columns: {display_cols}")
                            
                            if display_cols:
                                st.dataframe(unmatched_df[display_cols], use_container_width=True)
                    else:
                        st.warning("Results data doesn't contain matching information.")
                
                # Download button for results - placed at bottom of page
                st.download_button(
                    "Download Results CSV",
                    generate_download_link(st.session_state.results),
                    "circles_results.csv",
                    "text/csv",
                    key='download-csv'
                )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        if 'debug_mode' in st.session_state.config and st.session_state.config['debug_mode']:
            st.exception(e)

# Define callback for the Match tab
def match_tab_callback():
    st.subheader("Upload Participant Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV with participant data", 
        type=["csv", "xlsx"],
        help="Upload a CSV file with participant data following the required format"
    )
    
    if uploaded_file is not None:
        process_uploaded_file(uploaded_file)

# Use the match tab callback directly
match_tab_callback_func = match_tab_callback

if __name__ == "__main__":
    main()
