import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import os
from modules.data_loader import load_data, validate_data
from modules.data_processor import process_data, normalize_data
from modules.optimizer import run_matching_algorithm
try:
    from modules.circle_splitter import split_large_circles
except Exception as e:
    print(f"Error importing circle_splitter: {str(e)}")
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
from utils.feature_flags import initialize_feature_flags, set_flag

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
        'debug_mode': True,  # CRITICAL FIX: Force debug mode on to help diagnose compatibility issues
        'min_circle_size': 5,
        'existing_circle_handling': 'optimize',  # Always use optimize mode (no UI option to change this)
        'optimization_weight_location': 3,
        'optimization_weight_time': 2,
        'enable_host_requirement': True
    }

# Initialize feature flags
initialize_feature_flags()

# Enable the feature flags UI and key metadata features by default
set_flag('enable_feature_flags_ui', True)
set_flag('use_optimizer_metadata', True)  # Use optimizer-generated metadata
set_flag('enable_metadata_validation', True)  # Enable validation in Debug tab
set_flag('use_standardized_member_lists', True)  # Ensure consistent member list format
set_flag('use_standardized_host_status', True)  # Normalize host status values

def main():
    st.title("CirclesTool2")
    st.write("GSB Alumni Circle Matching Tool")
    
    # Create tabs for navigation, moved Demographics after Match per user request
    # Removed East Bay Debug tab to focus on Seattle testing
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Match", "Demographics", "Details", "Debug", "Split Test"])
    
    # Add circle splitting test tab for debugging
    with tab5:
        st.subheader("Circle Splitting Test")
        st.write("This tab is for testing the circle splitting functionality directly.")
        
        if st.button("Test Circle Splitting"):
            test_circle_splitting()
    
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
    
    # Always use 'optimize' mode for handling existing circles as requested
    print(f"🔄 Using 'optimize' circle handling mode (fixed setting)")
    
    # Force the config to use optimize mode regardless of what's in session state
    st.session_state.config['existing_circle_handling'] = 'optimize'
    
    # Update other config parameters from UI
    st.session_state.config['optimization_weight_location'] = st.session_state.get('location_weight', 5.0)
    st.session_state.config['optimization_weight_time'] = st.session_state.get('time_weight', 5.0)
    
    try:
        with st.spinner("Running matching algorithm..."):
            start_time = time.time()
            
            # Use the original data without any test participants
            # Run the matching algorithm with enhanced return values for debugging
            results, matched_circles, unmatched_participants = run_matching_algorithm(
                st.session_state.processed_data,
                st.session_state.config
            )
            
            # Add extensive diagnostic logging to understand data structure
            print("\n🔬🔬🔬 DETAILED RESULT ANALYSIS 🔬🔬🔬")
            print(f"Raw results length: {len(results)}")
            
            # Check for test participants that might be inflating counts
            # First, check if results is a DataFrame or a list of dictionaries
            test_participants = []
            if isinstance(results, pd.DataFrame):
                # If it's a DataFrame, use proper DataFrame filtering
                if 'Encoded ID' in results.columns:
                    mask = results['Encoded ID'].astype(str).str.startswith('99999')
                    test_participants = results[mask]
                    print(f"⚠️ FOUND {len(test_participants)} TEST PARTICIPANTS in results (DataFrame)")
                    if not test_participants.empty:
                        print(f"  First test participant ID: {test_participants.iloc[0]['Encoded ID']}")
            else:
                # If it's a list of dictionaries, use the original approach
                try:
                    test_participants = [r for r in results if isinstance(r, dict) and str(r.get('Encoded ID', '')).startswith('99999')]
                    if test_participants:
                        print(f"⚠️ FOUND {len(test_participants)} TEST PARTICIPANTS in results (List)")
                        print(f"  First test participant: {test_participants[0].get('Encoded ID', 'Unknown')}")
                except Exception as e:
                    print(f"⚠️ Error when checking for test participants: {str(e)}")
                    print(f"  Type of results: {type(results)}")
                    print(f"  Sample item type: {type(results[0]) if results and len(results) > 0 else 'No items'}")
            
            # Check for duplicate Encoded IDs
            if 'Encoded ID' in results.columns:
                total_ids = len(results['Encoded ID'])
                unique_ids = len(results['Encoded ID'].unique())
                print(f"Total IDs: {total_ids}, Unique IDs: {unique_ids}")
                
                if total_ids > unique_ids:
                    print(f"⚠️ FOUND {total_ids - unique_ids} DUPLICATE IDs in results")
                    print("🛠️ Fixing duplicates in results DataFrame")
                    
                    # De-duplicate the results DataFrame
                    results = results.drop_duplicates(subset=['Encoded ID'], keep='first')
                    print(f"✅ After de-duplication: {results.shape[0]} participants (was {total_ids})")
            
            # Count matched vs unmatched
            if 'proposed_NEW_circles_id' in results.columns:
                matched_in_results = len(results[results['proposed_NEW_circles_id'] != 'UNMATCHED'])
                unmatched_in_results = len(results[results['proposed_NEW_circles_id'] == 'UNMATCHED'])
                print(f"From results DataFrame - Matched: {matched_in_results}, Unmatched: {unmatched_in_results}")
            
            # Check the unmatched_participants parameter
            print(f"Unmatched participants parameter length: {len(unmatched_participants)}")
            
            # Check matched circles
            print(f"Matched circles length: {len(matched_circles)}")
            
            # Calculate total expected participants
            total_in_circles = sum(circle.get('member_count', 0) if isinstance(circle, dict) else 0 
                                  for circle in matched_circles)
            print(f"Total participants in all circles (member_count): {total_in_circles}")
            
            # Compare with original data
            original_participant_count = len(st.session_state.processed_data)
            print(f"Original data participant count: {original_participant_count}")
            
            # Properly handle test_participants which could be a DataFrame or a list
            test_count = 0
            if isinstance(test_participants, pd.DataFrame):
                test_count = len(test_participants) if not test_participants.empty else 0
            elif isinstance(test_participants, list):
                test_count = len(test_participants)
            elif hasattr(test_participants, '__len__'):  # Handle any other iterable
                test_count = len(test_participants)
            print(f"Test participant adjustment: -{test_count}")
            
            # CRITICAL: Apply centralized metadata fixes to results
            print("\n🔧 APPLYING CENTRALIZED METADATA FIXES AT SOURCE")
            try:
                # First try the new approach using CircleMetadataManager
                from utils.circle_metadata_manager import initialize_or_update_manager
                print("  Using CircleMetadataManager for comprehensive metadata management")
                
                # Add metadata_source column to matched_circles if it doesn't exist already
                if isinstance(matched_circles, pd.DataFrame) and not matched_circles.empty:
                    # Check if we need to add metadata_source column
                    if 'metadata_source' not in matched_circles.columns:
                        print("  Adding metadata_source column to matched_circles")
                        matched_circles['metadata_source'] = 'optimizer'
                    
                    # Debug what's in the matched_circles DataFrame
                    print(f"  Number of matched circles: {len(matched_circles)}")
                    if len(matched_circles) > 0 and 'circle_id' in matched_circles.columns:
                        print(f"  Sample circle IDs: {matched_circles['circle_id'].head(5).tolist()}")
                
                # Get feature flag
                from utils.feature_flags import get_flag
                use_optimizer_metadata = get_flag('use_optimizer_metadata')
                if use_optimizer_metadata:
                    print("  ✅ Using optimizer metadata for circle reconstruction")
                    
                # Initialize/update the circle metadata manager with optimizer results
                circle_manager = initialize_or_update_manager(
                    st.session_state,
                    optimizer_circles=matched_circles,
                    results_df=results
                )
                
                # Log success and details of the manager
                if circle_manager:
                    print(f"  ✅ Successfully initialized/updated CircleMetadataManager")
                    print(f"  Circle count: {len(circle_manager.get_all_circles())} circles")
                    
                    # Special debug for target circles
                    for test_circle in ['IP-BOS-04', 'IP-BOS-05']:
                        circle_data = circle_manager.get_circle_data(test_circle)
                        if circle_data:
                            print(f"\n  TARGET CIRCLE {test_circle} STATUS:")
                            for key in ['always_hosts', 'sometimes_hosts', 'max_additions', 'member_count']:
                                print(f"    {key}: {circle_data.get(key, 'Not Found')}")
                else:
                    print("  ⚠️ Could not initialize CircleMetadataManager")
                    
                # Legacy fallback in case we need it
                try:
                    from utils.metadata_manager import fix_participant_metadata_in_results
                    # Apply the fixes to the results dataframe
                    fixed_results = fix_participant_metadata_in_results(results)
                    # Update with the fixed data
                    results = fixed_results
                    print("  ✅ Applied legacy metadata fixes to participant results")
                except Exception as e:
                    print(f"  ℹ️ Legacy metadata fixes not applied: {str(e)}")
            except Exception as e:
                print(f"⚠️ Error applying metadata fixes: {str(e)}")
                print("  Continuing with original results")
                
            # Store results in session state
            st.session_state.results = results
            st.session_state.unmatched_participants = unmatched_participants
            st.session_state.exec_time = time.time() - start_time
            
            # ENHANCED APPROACH: Use CircleMetadataManager for consistent circle data management
            # Note: We already initialized this earlier, so this code is now redundant
            # Simply retrieve the existing manager from session state
            
            # Check if the manager was properly initialized
            from utils.circle_metadata_manager import get_manager_from_session_state
            
            print("\n🔄 VERIFYING CIRCLE METADATA MANAGER INITIALIZATION")
            manager = get_manager_from_session_state(st.session_state)
            
            if manager:
                # For backward compatibility, we still set matched_circles in session state
                # but the CircleMetadataManager is now the authoritative source
                circles_df = manager.get_circles_dataframe()
                st.session_state.matched_circles = circles_df
                print(f"  ✅ CircleMetadataManager is active with {len(circles_df)} circles")
                
                # Add special debug for target circles
                for test_id in ['IP-BOS-04', 'IP-BOS-05']:
                    circle_data = manager.get_circle_data(test_id)
                    if circle_data:
                        print(f"\n🔍 VALIDATION FOR {test_id} FROM METADATA MANAGER:")
                        for key in ['max_additions', 'always_hosts', 'sometimes_hosts', 'member_count']:
                            print(f"  {key}: {circle_data.get(key, 'Not Found')}")
            else:
                # The manager was not initialized earlier, create it now
                print("  ⚠️ CircleMetadataManager not found in session state, reinitializing")
                from utils.circle_metadata_manager import initialize_or_update_manager
                
                # Initialize metadata manager with circle data and results
                manager = initialize_or_update_manager(st.session_state, matched_circles, results)
                if manager:
                    circles_df = manager.get_circles_dataframe()
                    st.session_state.matched_circles = circles_df
                    print(f"  ✅ Successfully initialized CircleMetadataManager with {len(circles_df)} circles")
                else:
                    st.session_state.matched_circles = matched_circles
                    print("  ⚠️ Failed to initialize CircleMetadataManager, falling back to direct storage")
                    # Log detailed types for debugging
                    print(f"  matched_circles type: {type(matched_circles)}")
                    print(f"  results type: {type(results)}")
            
            # Debug verification of max_additions values
            print("\n🔍 VERIFYING MAX ADDITIONS: Checking that values are correctly preserved")
            if 'circle_manager' in st.session_state:
                sample_circles = st.session_state.circle_manager.get_all_circles()[:3]
                for circle in sample_circles:
                    print(f"  Circle {circle['circle_id']}: max_additions={circle.get('max_additions', 'N/A')}")
            elif 'matched_circles' in st.session_state:
                if hasattr(st.session_state.matched_circles, 'head'):
                    sample_df = st.session_state.matched_circles.head(3)
                    if 'max_additions' in sample_df.columns:
                        for _, row in sample_df.iterrows():
                            print(f"  Circle {row['circle_id']}: max_additions={row.get('max_additions', 'N/A')}")
                    else:
                        print("  ⚠️ max_additions column not found in matched_circles DataFrame")
            
            # Debug verification after storing in session state
            if 'results' in st.session_state and isinstance(st.session_state.results, pd.DataFrame):
                results_df = st.session_state.results
                if 'proposed_NEW_Subregion' in results_df.columns:
                    unknown_count = results_df[results_df['proposed_NEW_Subregion'] == 'Unknown'].shape[0]
                    total_count = results_df.shape[0]
                    print(f"VERIFICATION: {unknown_count}/{total_count} Unknown subregions in session state results")
                    
                if 'proposed_NEW_DayTime' in results_df.columns:
                    unknown_count = results_df[results_df['proposed_NEW_DayTime'] == 'Unknown'].shape[0]
                    total_count = results_df.shape[0]
                    print(f"VERIFICATION: {unknown_count}/{total_count} Unknown meeting times in session state results")
                
            # ENHANCED APPROACH: Check if there was an issue initializing the CircleMetadataManager earlier
            # If so, try again with the stored data
            if 'circle_manager' not in st.session_state and 'matched_circles' in st.session_state:
                print("\n🔄 INITIALIZING METADATA MANAGER (SECOND ATTEMPT): Using stored circle data")
                from utils.circle_metadata_manager import initialize_or_update_manager
                
                # Initialize metadata manager with circle data and results
                manager = initialize_or_update_manager(st.session_state)
                if manager:
                    print(f"  ✅ Successfully initialized CircleMetadataManager on second attempt")
                    
                    # Verify max_additions values
                    print("\n🔍 MAX ADDITIONS VERIFICATION (2nd check):")
                    sample_circles = manager.get_all_circles()[:3]
                    for circle in sample_circles:
                        print(f"  Circle {circle['circle_id']}: max_additions={circle.get('max_additions', 'N/A')}")
                else:
                    print("  ⚠️ Failed to initialize CircleMetadataManager on second attempt")
            
            # Calculate standard statistics with our helper function
            from utils.helpers import calculate_matching_statistics
            match_stats = calculate_matching_statistics(results, matched_circles)
            st.session_state.match_statistics = match_stats
            
            # We no longer show the warning about filtered participants
            # The statistics are still accurate but we don't display the message
            
            # Calculate and store diversity score immediately after optimization
            from modules.ui_components import calculate_total_diversity_score
            total_diversity_score = calculate_total_diversity_score(matched_circles, results)
            st.session_state.total_diversity_score = total_diversity_score
            print(f"DEBUG - Calculated diversity score immediately after optimization: {total_diversity_score}")
            
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
                        print("ℹ️ No logs found in file backup")
                        # Initialize with empty logs - no test data generation
                        st.session_state.circle_eligibility_logs = {}
                        
                        # Show a warning message to the user
                        st.warning("No circle eligibility data available. This usually means the optimization process didn't generate circle data properly. Please try running the optimization again.")
                except Exception as e:
                    print(f"❌ Error during file operations: {str(e)}")
                    st.session_state.circle_eligibility_logs = {}
                    st.warning("Unable to load circle data. Please try running the optimization again.")
            
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
            
            # ENHANCED DIAGNOSTICS: Print detailed info about the input data
            print("\n🔬🔬🔬 SUPER DETAILED DATA DIAGNOSTICS IN PROCESS_UPLOADED_FILE 🔬🔬🔬")
            print(f"🔬 DataFrame shape: {df.shape}")
            print(f"🔬 DataFrame columns: {df.columns.tolist()}")
            
            # Count participants by status for debugging
            if 'Status' in df.columns:
                status_counts = df['Status'].value_counts().to_dict()
                st.session_state.status_counts = status_counts
                print(f"🔬 Status counts: {status_counts}")
            else:
                print("🔬 'Status' column not found")
                
            # Check circle ID column
            circle_columns = ['Current_Circle_ID', 'current_circles_id', 'Current Circle ID']
            found_col = None
            for col in circle_columns:
                if col in df.columns:
                    found_col = col
                    break
                    
            if found_col:
                print(f"🔬 Found circle ID column: {found_col}")
                
                # Count non-null values
                non_null_count = df[~df[found_col].isna()].shape[0]
                print(f"🔬 Participants with non-null {found_col}: {non_null_count}")
                
                # Check CURRENT-CONTINUING participants with circle IDs
                if 'Status' in df.columns:
                    continuing = df[df['Status'] == 'CURRENT-CONTINUING']
                    print(f"🔬 CURRENT-CONTINUING participants: {len(continuing)}")
                    
                    with_circles = continuing[~continuing[found_col].isna()]
                    print(f"🔬 CURRENT-CONTINUING with circle IDs: {len(with_circles)}")
                    
                    if len(with_circles) > 0:
                        unique_circles = with_circles[found_col].unique()
                        print(f"🔬 Unique circle IDs: {len(unique_circles)}")
                        print(f"🔬 First 10 circle IDs: {list(unique_circles)[:10]}")
                        
                        # Get circle member counts
                        circle_counts = with_circles[found_col].value_counts()
                        print(f"🔬 Circle member counts (top 10):")
                        for circle, count in circle_counts.head(10).items():
                            print(f"   {circle}: {count} members")
                            
                        # Check for any problematic circle patterns
                        problematic_patterns = ['IP-TEST', 'IP-NEW-TES']
                        for pattern in problematic_patterns:
                            problem_circles = [c for c in unique_circles if pattern in c]
                            if problem_circles:
                                print(f"🚨 WARNING: Found {len(problem_circles)} circles with test pattern '{pattern}'")
                                print(f"   Circle IDs: {problem_circles}")
                                print(f"   These may be test circles and should be removed from production data!")
            else:
                print("🔬 No valid circle ID column found")
                
            print("🔬🔬🔬 END OF SUPER DETAILED DATA DIAGNOSTICS 🔬🔬🔬\n")
            
            # Display validation errors if any
            if len(validation_errors) > 0:
                st.warning(f"Found {len(validation_errors)} validation issues:")
                for error in validation_errors[:5]:  # Show first 5 errors
                    st.write(f"- {error}")
                if len(validation_errors) > 5:
                    st.write(f"...and {len(validation_errors) - 5} more issues.")
            
            # Display filtered status counts and duplicate IDs
            if 'status_filter_counts' in st.session_state:
                counts = st.session_state.status_filter_counts
                filter_message = []
                filter_message.append(f"{counts.get('not_continuing', 0)} NOT Continuing records filtered")
                filter_message.append(f"{counts.get('moving_out', 0)} MOVING OUT records filtered")
                if len(deduplication_messages) > 0:
                    filter_message.append(f"{len(deduplication_messages)} duplicate Encoded IDs fixed")
                if True:  # Always show the message box
                    st.warning(" • ".join(filter_message))
            
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
            # Always use 'optimize' mode (no UI option to change this)
            st.session_state.config['existing_circle_handling'] = 'optimize'
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
                    
                    # EXTENSIVE DIAGNOSTIC LOGGING FOR PARTICIPANT COUNT ISSUE
                    print("\n🔍🔍🔍 MATCH COUNT DIAGNOSTICS 🔍🔍🔍")
                    print(f"Raw results_df shape: {results_df.shape}")
                    
                    # Check for duplicate IDs which could be inflating the count
                    if 'Encoded ID' in results_df.columns:
                        total_ids = len(results_df['Encoded ID'])
                        unique_ids = len(results_df['Encoded ID'].unique())
                        print(f"Total IDs: {total_ids}, Unique IDs: {unique_ids}")
                        if total_ids > unique_ids:
                            print(f"⚠️ FOUND {total_ids - unique_ids} DUPLICATE IDs!")
                            # Show examples of duplicates
                            duplicate_mask = results_df.duplicated(subset=['Encoded ID'], keep=False)
                            duplicates = results_df[duplicate_mask]
                            if len(duplicates) > 0:
                                print(f"Examples of duplicated IDs:")
                                dup_examples = duplicates['Encoded ID'].unique()[:5]
                                for dup_id in dup_examples:
                                    dup_rows = results_df[results_df['Encoded ID'] == dup_id]
                                    print(f"  ID {dup_id} appears {len(dup_rows)} times")
                                    if len(dup_rows) > 1:
                                        for i, (_, row) in enumerate(dup_rows.iterrows()):
                                            circle_id = row.get('proposed_NEW_circles_id', 'N/A')
                                            status = row.get('Status', 'N/A')
                                            print(f"    Instance {i+1}: Circle={circle_id}, Status={status}")
                            
                            # CRITICAL FIX: De-duplicate the results dataframe to fix inflated participant count
                            print("🛠️ APPLYING FIX: Removing duplicate Encoded IDs from results dataframe")
                            # Keep the first occurrence of each Encoded ID
                            results_df = results_df.drop_duplicates(subset=['Encoded ID'], keep='first')
                            print(f"✅ After de-duplication: {results_df.shape[0]} participants (was {total_ids})")
                            
                            # Update the session state with the de-duplicated dataframe
                            st.session_state.results = results_df
                            # Recalculate total participants
                            total_participants = len(results_df)
                    
                    # FIXED: More accurate calculation of matched participants
                    # Only count non-empty circle IDs to avoid counting filtered records
                    matched_count = 0
                    unmatched_count = 0
                    
                    if 'proposed_NEW_circles_id' in results_df.columns:
                        # Check all values for diagnostics
                        circle_values = results_df['proposed_NEW_circles_id'].value_counts().to_dict()
                        print(f"Circle ID values count:")
                        
                        # Group by UNMATCHED, nan/null, and actual circles
                        unmatched_count = circle_values.get('UNMATCHED', 0)
                        null_count = results_df['proposed_NEW_circles_id'].isna().sum()
                        circle_count = sum(v for k, v in circle_values.items() 
                                           if k != 'UNMATCHED' and not (isinstance(k, str) and k.strip() == '') and
                                           not (hasattr(pd.isna(k), '__iter__') and pd.isna(k).all() if hasattr(pd.isna(k), '__iter__') else pd.isna(k)))
                        
                        print(f"  Assigned to circles: {circle_count}")
                        print(f"  UNMATCHED: {unmatched_count}")
                        print(f"  Null/NaN: {null_count}")
                        
                        # Filter out any null or empty values
                        notna_mask = results_df['proposed_NEW_circles_id'].notna()
                        valid_circle_mask = notna_mask & (results_df['proposed_NEW_circles_id'] != 'UNMATCHED')
                        matched_count = len(results_df[valid_circle_mask])
                        unmatched_count = len(results_df[results_df['proposed_NEW_circles_id'] == 'UNMATCHED'])
                        
                        # Compare with actual circle data
                        circles_participant_count = 0
                        if 'matched_circles' in st.session_state:
                            circles = st.session_state.matched_circles
                            
                            # DEBUG: Check the type and validity of circles
                            print(f"\n🔍 CIRCLES DEBUG: matched_circles type is {type(circles)}")
                            
                            # Add safety checks for None and empty values
                            if circles is None:
                                print("⚠️ WARNING: matched_circles is None. Skipping member count calculation.")
                            elif isinstance(circles, pd.DataFrame) and circles.empty:
                                print("⚠️ WARNING: matched_circles DataFrame is empty. Skipping member count calculation.")
                            elif isinstance(circles, list) and len(circles) == 0:
                                print("⚠️ WARNING: matched_circles list is empty. Skipping member count calculation.")
                            else:
                                # Handle different collection types
                                if isinstance(circles, pd.DataFrame):
                                    # DataFrame direct access
                                    print(f"🔍 Processing DataFrame with {len(circles)} rows")
                                    if 'member_count' in circles.columns:
                                        circles_participant_count = circles['member_count'].sum()
                                    print(f"🔍 Circle member count from DataFrame: {circles_participant_count}")
                                    
                                elif isinstance(circles, list) or hasattr(circles, '__iter__'):
                                    # Safe iteration for any iterable
                                    print(f"🔍 Processing {type(circles)} with {len(circles) if hasattr(circles, '__len__') else 'unknown'} items")
                                    iterator_count = 0
                                    
                                    # Defensive programming - ensure circles is valid before iteration
                                    if circles is None:
                                        print("⚠️ CRITICAL: circles was None during iteration. Using empty list.")
                                        safe_circles = []
                                    else:
                                        # Ensure we have something iterable
                                        try:
                                            # Fast check for common collections
                                            if isinstance(circles, (list, tuple, pd.DataFrame)):
                                                safe_circles = circles
                                            else:
                                                # Try to convert to a list explicitly
                                                safe_circles = list(circles)
                                        except Exception as e:
                                            print(f"⚠️ CRITICAL: Error converting circles to iterable: {str(e)}")
                                            safe_circles = []
                                    
                                    # Handle different types properly with the safe iterable
                                    try:
                                        for circle in safe_circles:
                                            iterator_count += 1
                                            if isinstance(circle, dict):
                                                # Dictionary circle
                                                circles_participant_count += circle.get('member_count', 0)
                                            elif isinstance(circle, pd.DataFrame):
                                                # DataFrame circle
                                                if 'member_count' in circle.columns:
                                                    circles_participant_count += circle['member_count'].sum()
                                            elif isinstance(circle, str):
                                                # String entries (circle IDs) - can't get member count directly
                                                print(f"⚠️ Found string circle entry: {circle}")
                                            else:
                                                # Other types - log for debugging
                                                print(f"⚠️ Unknown circle type: {type(circle)}")
                                    except Exception as e:
                                        print(f"⚠️ Error during circle iteration: {str(e)}")
                                        # Continue without failing
                                    
                                    print(f"🔍 Iterated through {iterator_count} items")
                                else:
                                    # Unknown, just log it
                                    print(f"⚠️ matched_circles is unexpected type: {type(circles)}")
                                
                                print(f"Total participants in matched_circles: {circles_participant_count}")
                        else:
                            print("⚠️ WARNING: 'matched_circles' not found in session state")
                    
                    # Check if we have test participants and remove them
                    if 'Encoded ID' in results_df.columns:
                        try:
                            # Identify test participants (IDs starting with 99999)
                            mask = results_df['Encoded ID'].astype(str).str.startswith('99999')
                            test_participants_df = results_df[mask]
                            
                            if len(test_participants_df) > 0:
                                print(f"⚠️ FOUND {len(test_participants_df)} TEST PARTICIPANTS that might be inflating counts")
                                print(f"   Test participant IDs: {test_participants_df['Encoded ID'].tolist()}")
                                
                                # Remove test participants from results
                                results_df = results_df[~mask]
                                print(f"✅ FILTERED OUT {len(test_participants_df)} test participants from results")
                                print(f"   New result count: {len(results_df)} (was {len(results_df) + len(test_participants_df)})")
                                
                                # Update the session state with filtered results
                                st.session_state.results = results_df
                                # Recalculate total participants
                                total_participants = len(results_df)
                        except Exception as e:
                            print(f"⚠️ Error filtering test participants: {str(e)}")
                            print(f"Type of Encoded ID column: {results_df['Encoded ID'].dtype}")
                    
                    # Call the standardized statistics calculation function
                    from utils.helpers import calculate_matching_statistics
                    
                    # Calculate standardized statistics
                    circles_df = st.session_state.matched_circles
                    match_stats = calculate_matching_statistics(results_df, circles_df)
                    
                    # Store the statistics in session state for use throughout the app
                    st.session_state.match_statistics = match_stats
                    
                    # Log the calculated statistics
                    print("\n🔍🔍🔍 STANDARDIZED STATISTICS CALCULATION 🔍🔍🔍")
                    print(f"Total participants: {match_stats['total_participants']}")
                    print(f"Matched participants: {match_stats['matched_participants']}")
                    print(f"Unmatched participants: {match_stats['unmatched_participants']}")
                    print(f"Match rate: {match_stats['match_rate']:.1f}%")
                    print(f"Total circles: {match_stats['total_circles']}")
                    
                    # Log comparison between calculation methods
                    if 'details_matched_count' in match_stats:
                        print(f"\nComparison between calculation methods:")
                        print(f"Match page method (results DataFrame): {match_stats['matched_participants']}")
                        print(f"Details page method (circle member counts): {match_stats['details_matched_count']}")
                        print(f"Discrepancy: {match_stats['match_discrepancy']}")
                    
                    # Log details about test circles if they exist
                    if 'test_circles' in match_stats:
                        print(f"\nTest circles found: {match_stats['test_circles']}")
                        print(f"Adjusted statistics (excluding test circles):")
                        for key, value in match_stats['adjusted_statistics'].items():
                            print(f"  {key}: {value}")
                            
                    # ENHANCED CSV DIAGNOSTICS: Compare with what will appear in the CSV
                    print("\n🔍🔍🔍 COMPARING UI STATS VS CSV EXPORT 🔍🔍🔍")
                    
                    # Capture the list of matched participants that contribute to the UI count
                    valid_circle_mask = (results_df['proposed_NEW_circles_id'].notna()) & (results_df['proposed_NEW_circles_id'] != 'UNMATCHED')
                    ui_matched_ids = results_df[valid_circle_mask]['Encoded ID'].tolist()
                    
                    # Store this for comparison when CSV is downloaded
                    st.session_state.ui_matched_ids = ui_matched_ids
                    
                    # Generate a test CSV (same process used in the download button)
                    from utils.helpers import generate_download_link, get_valid_participants
                    # Filter out participants with null Encoded IDs for consistent counting
                    filtered_results = get_valid_participants(results_df)
                    print(f"🔍 Test CSV: Using {len(filtered_results)} valid participants with non-null Encoded IDs")
                    test_csv_content = generate_download_link(filtered_results)
                    
                    # The CSV generation returns the full CSV content, we'd need to parse it back to get the IDs
                    # Instead of parsing the CSV, let's just store both for comparison
                    print(f"Original UI Matched Count: {len(ui_matched_ids)}")
                    print(f"First 5 Matched IDs in UI: {ui_matched_ids[:5] if len(ui_matched_ids) >= 5 else ui_matched_ids}")
                    
                    # Capture any participants with null Encoded IDs that would be filtered in CSV
                    null_id_matched = results_df[valid_circle_mask & results_df['Encoded ID'].isna()]
                    if len(null_id_matched) > 0:
                        print(f"⚠️ Found {len(null_id_matched)} matched participants with NULL Encoded ID")
                        for idx, row in null_id_matched.iterrows():
                            circle_id = row['proposed_NEW_circles_id']
                            status = row.get('Status', 'Unknown')
                            participant_id = row.get('participant_id', 'Unknown')
                            
                            print(f"  NULL ENCODED ID ENTRY DETAILS (THIS IS WHAT YOU'RE LOOKING FOR):")
                            print(f"  Record index: {idx}")
                            print(f"  Circle: {circle_id}, Status: {status}, participant_id: {participant_id}")
                            
                            # Show key information about this phantom participant
                            location_score = row.get('location_score', 'Unknown')
                            time_score = row.get('time_score', 'Unknown')
                            total_score = row.get('total_score', 'Unknown')
                            region = row.get('region', 'Unknown')
                            
                            print(f"  Key scores: location={location_score}, time={time_score}, total={total_score}")
                            print(f"  Region: {region}")
                            
                            # Let's dump the entire row to help identify this record
                            print(f"  FULL RECORD DUMP:")
                            non_null_attrs = {}
                            for col, val in row.items():
                                if pd.notna(val) and not col.startswith('Unnamed'):
                                    non_null_attrs[col] = val
                                    
                            # Print the attribute keys first
                            print(f"  All non-null attribute keys: {list(non_null_attrs.keys())}")
                            
                            # Then print each value
                            for col, val in non_null_attrs.items():
                                print(f"  - {col}: {val}")
                                
                            # Update user about the phantom participant and show it on screen
                            st.session_state['phantom_participant_id'] = participant_id
                            st.session_state['phantom_circle'] = circle_id
                            
                            # Create a special message at the top of the page
                            if 'show_phantom_info' not in st.session_state:
                                st.session_state['show_phantom_info'] = True
                            
                    # Add diagnostic mode to print all matched IDs for comparison
                    if st.session_state.get('config', {}).get('debug_mode', False):
                        print("All UI Matched IDs for comparison:")
                        print(ui_matched_ids)
                    
                    print("🔍🔍🔍 END STANDARDIZED STATISTICS 🔍🔍🔍\n")
                    
                    # IMPORTANT: For backward compatibility, set these variables that are used later in the code
                    # In the future, these should be refactored to use st.session_state.match_statistics directly
                    matched_count = match_stats['matched_participants']
                    unmatched_count = match_stats['unmatched_participants']
                    total_participants = match_stats['total_participants']
                    
                    # Create columns for the metrics (now 3 columns instead of 4)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None:
                            st.metric("Circles Created", match_stats['total_circles'])
                    
                    with col2:
                        # Use the standardized matched count
                        st.metric("Participants Matched", match_stats['matched_participants'])
                        
                    with col3:
                        # Use the standardized match rate
                        st.metric("Match Success Rate", f"{match_stats['match_rate']:.1f}%")
                
                # Create size histogram before the composition table
                st.subheader("Circle Size Distribution")
                if 'matched_circles' in st.session_state and st.session_state.matched_circles is not None and 'member_count' in st.session_state.matched_circles.columns:
                    circles_df = st.session_state.matched_circles
                    
                    # Calculate size counts 
                    size_counts = circles_df['member_count'].value_counts().sort_index()
                    
                    # Create DataFrame for plotting
                    size_df = pd.DataFrame({
                        'Circle Size': size_counts.index,
                        'Number of Circles': size_counts.values
                    })
                    
                    # Create histogram using plotly with Stanford cardinal red color
                    fig = px.bar(
                        size_df,
                        x='Circle Size',
                        y='Number of Circles',
                        title='Distribution of Circle Sizes',
                        text='Number of Circles',  # Display count values on bars
                        color_discrete_sequence=['#8C1515']  # Stanford Cardinal red
                    )
                    
                    # Customize layout
                    fig.update_traces(textposition='outside')
                    fig.update_layout(
                        xaxis=dict(
                            title="Number of Members",
                            tickmode='linear',
                            dtick=1  # Force integer labels
                        ),
                        yaxis_title="Number of Circles"
                    )
                    
                    # Show the plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show summary statistics
                    avg_size = circles_df['member_count'].mean()
                    median_size = circles_df['member_count'].median()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Circle Size", f"{avg_size:.1f}")
                    with col2:
                        st.metric("Median Circle Size", f"{median_size:.0f}")

                st.subheader("Circle Composition")
                
                # Display circle table with specified columns
                if ('matched_circles' in st.session_state and 
                    st.session_state.matched_circles is not None and 
                    (isinstance(st.session_state.matched_circles, list) or 
                     not (hasattr(st.session_state.matched_circles, 'empty') and st.session_state.matched_circles.empty))):
                    
                    # Get the data - handle the case where matched_circles might be a list of dictionaries
                    if isinstance(st.session_state.matched_circles, pd.DataFrame):
                        circles_df = st.session_state.matched_circles.copy()
                    elif isinstance(st.session_state.matched_circles, list):
                        # Convert list of dictionaries to DataFrame
                        try:
                            circles_df = pd.DataFrame(st.session_state.matched_circles)
                        except Exception as e:
                            st.warning(f"Could not convert circles to DataFrame: {str(e)}")
                            print(f"Error converting matched_circles to DataFrame: {str(e)}")
                            circles_df = pd.DataFrame()  # Empty DataFrame as fallback
                    else:
                        # Unknown type
                        st.warning(f"Unexpected matched_circles type: {type(st.session_state.matched_circles)}")
                        circles_df = pd.DataFrame()  # Empty DataFrame as fallback
                    
                    # SYSTEM UPGRADE: Using the centralized metadata manager for comprehensive fixing
                    from utils.circle_metadata_manager import get_manager_from_session_state, initialize_or_update_manager
                    import logging
                    
                    # Get the metadata manager from session state
                    print("\n🔄 GETTING CIRCLE DATA FROM METADATA MANAGER: Authoritative single source of truth")
                    metadata_manager = get_manager_from_session_state(st.session_state)
                    
                    # Check if we got a valid manager
                    if metadata_manager:
                        # Replace the circles_df with the one from the manager to ensure consistency
                        circles_df = metadata_manager.get_circles_dataframe()
                        print(f"  ✅ Successfully retrieved circle data from metadata manager: {len(circles_df)} circles")
                        
                        # CRITICAL FIX: Target circle diagnostics
                        for test_id in ['IP-BOS-04', 'IP-BOS-05']:
                            circle_data = metadata_manager.get_circle_data(test_id)
                            if circle_data:
                                print(f"\n🔍 TEST CIRCLE {test_id} VALIDATION FROM METADATA MANAGER:")
                                for key in ['max_additions', 'always_hosts', 'sometimes_hosts', 'member_count', 'new_members', 'continuing_members']:
                                    print(f"  {key}: {circle_data.get(key, 'Not Found')}")
                    else:
                        print("  ⚠️ WARNING: Could not get metadata manager from session state")
                    
                    print("\n🔍 CIRCLE COMPOSITION DEBUG: Using centralized metadata manager")
                    print(f"Circle DataFrame shape: {circles_df.shape if hasattr(circles_df, 'shape') else 'unknown'}")
                    if hasattr(circles_df, 'columns'):
                        print(f"Available columns: {list(circles_df.columns)}")
                        
                        # Count how many subregion and meeting_time values are "Unknown"
                        if 'subregion' in circles_df.columns:
                            unknown_subregions = circles_df[circles_df['subregion'] == 'Unknown'].shape[0]
                            total_subregions = circles_df.shape[0]
                            print(f"'Unknown' subregions: {unknown_subregions}/{total_subregions} ({unknown_subregions/total_subregions*100:.1f}%)")
                            # Log first 5 circles with Unknown subregion
                            if unknown_subregions > 0:
                                unknown_circles = circles_df[circles_df['subregion'] == 'Unknown']['circle_id'].tolist()[:5]
                                print(f"Sample circles with Unknown subregion: {unknown_circles}")
                        
                        if 'meeting_time' in circles_df.columns:
                            unknown_times = circles_df[circles_df['meeting_time'] == 'Unknown'].shape[0]
                            total_times = circles_df.shape[0]
                            print(f"'Unknown' meeting times: {unknown_times}/{total_times} ({unknown_times/total_times*100:.1f}%)")
                            # Log first 5 circles with Unknown meeting time
                            if unknown_times > 0:
                                unknown_circles = circles_df[circles_df['meeting_time'] == 'Unknown']['circle_id'].tolist()[:5]
                                print(f"Sample circles with Unknown meeting time: {unknown_circles}")
                    
                    # CRITICAL DEBUG: Log a detailed sample of raw circle data
                    if hasattr(circles_df, 'iterrows'):
                        print("\n🔍 SAMPLE CIRCLE DATA FOR FIRST 3 CIRCLES WITH UNKNOWN SUBREGION OR MEETING TIME:")
                        sample_count = 0
                        for _, row in circles_df.iterrows():
                            if ((row.get('subregion', '') == 'Unknown' or row.get('meeting_time', '') == 'Unknown') 
                                and sample_count < 3):
                                sample_count += 1
                                circle_id = row.get('circle_id', 'Unknown')
                                region = row.get('region', 'Unknown')
                                subregion = row.get('subregion', 'Unknown')
                                meeting_time = row.get('meeting_time', 'Unknown')
                                print(f"Circle {circle_id} (Region: {region})")
                                print(f"  Subregion: {subregion}")
                                print(f"  Meeting Time: {meeting_time}")
                                # Log all non-NA attributes to help with debugging
                                print(f"  All data attributes:")
                                for col, val in row.items():
                                    # Handle array-like pd.isna() values by using .all() to check if all elements are NaN
                                    is_valid_value = True
                                    try:
                                        # For array-like values, check if any are not NaN
                                        if hasattr(pd.isna(val), '__iter__'):
                                            is_valid_value = not pd.isna(val).all()
                                        else:
                                            is_valid_value = not pd.isna(val)
                                    except Exception as e:
                                        print(f"Warning: Error checking NaN for {col}: {str(e)}")
                                        # Default to showing the value if there's an error
                                        is_valid_value = True
                                        
                                    if is_valid_value and col not in ['circle_id', 'region', 'subregion', 'meeting_time']:
                                        print(f"    {col}: {val}")
                    
                    # COMPREHENSIVE FIX: Apply centralized metadata manager to fix Unknown values
                    # First, check the results_df columns to debug why previous fix attempts failed
                    if 'results' in st.session_state and st.session_state.results is not None:
                        results_df = st.session_state.results
                        
                        # Examine results columns to understand why previous fixes failed
                        print("\n🔍 DETAILED RESULTS DATA ANALYSIS:")
                        print(f"Results DataFrame shape: {results_df.shape}")
                        print(f"Results columns: {list(results_df.columns)}")
                        
                        # Check for specific columns that we expect to use for repairs
                        interesting_cols = [
                            'proposed_NEW_circles_id', 'assigned_circle', 'circle_id',
                            'proposed_NEW_Subregion', 'proposed_NEW_DayTime', 'Current_Subregion', 
                            'Current_Meeting_Time', 'Subregion', 'Meeting_Time'
                        ]
                        
                        found_cols = [col for col in interesting_cols if col in results_df.columns]
                        print(f"Found these interesting columns: {found_cols}")
                        
                        # Check what values exist in the identified columns
                        if found_cols:
                            print("Sample values from key columns:")
                            for col in found_cols[:5]:  # Show first 5 columns max
                                print(f"  Column '{col}':")
                                # Get unique non-null values
                                values = results_df[col].dropna().unique()
                                if len(values) > 0:
                                    # Show up to 5 sample values
                                    print(f"    Sample values: {values[:5]}")
                                else:
                                    print(f"    No non-null values found")
                        
                        # Now apply the comprehensive fix using our metadata manager
                        # Set up logging to capture detailed output
                        logging.basicConfig(level=logging.INFO)
                        
                        # CRITICAL FIX: Use our metadata manager for comprehensive fixing
                        if metadata_manager:
                            print("\n🔧 APPLYING COMPREHENSIVE METADATA FIX using CircleMetadataManager")
                            
                            # Update metadata from results
                            metadata_manager.results_df = results_df
                            
                            # Normalize and validate circle data
                            metadata_manager.normalize_metadata()
                            metadata_manager.validate_circles()
                            
                            # Get the updated data as a DataFrame
                            fixed_circles_df = metadata_manager.get_circles_dataframe()
                            print(f"  ✅ Successfully applied comprehensive metadata fix to {len(fixed_circles_df)} circles")
                        else:
                            print("  ⚠️ WARNING: Cannot apply comprehensive fix - no metadata manager available")
                            # Fallback to the original circles_df
                            fixed_circles_df = circles_df
                        
                        # Update the circles DataFrame with the fixed version
                        circles_df = fixed_circles_df
                        
                        # Final verification
                        if 'subregion' in circles_df.columns:
                            unknown_count = circles_df[circles_df['subregion'] == 'Unknown'].shape[0]
                            if unknown_count > 0:
                                print(f"⚠️ After fixes, still have {unknown_count} circles with Unknown subregion")
                            else:
                                print("✅ All circles now have valid subregion data!")
                                
                        if 'meeting_time' in circles_df.columns:
                            unknown_count = circles_df[circles_df['meeting_time'] == 'Unknown'].shape[0]
                            if unknown_count > 0:
                                print(f"⚠️ After fixes, still have {unknown_count} circles with Unknown meeting time")
                            else:
                                print("✅ All circles now have valid meeting time data!")
                    else:
                        print("⚠️ No results data available for metadata fixes")
                        
                    # If we still have unknowns after the comprehensive fix, log which circles remain problematic
                    if hasattr(circles_df, 'columns'):
                        remaining_unknown = []
                        if 'subregion' in circles_df.columns and 'meeting_time' in circles_df.columns:
                            still_unknown = circles_df[
                                (circles_df['subregion'] == 'Unknown') | 
                                (circles_df['meeting_time'] == 'Unknown')
                            ]
                            
                            if not still_unknown.empty:
                                print("\n⚠️ REMAINING PROBLEM CIRCLES:")
                                for _, row in still_unknown.iterrows():
                                    circle_id = row.get('circle_id', 'Unknown')
                                    region = row.get('region', 'Unknown')
                                    subregion = row.get('subregion', 'Unknown')
                                    meeting_time = row.get('meeting_time', 'Unknown')
                                    print(f"Circle {circle_id} (Region: {region})")
                                    if subregion == 'Unknown':
                                        print(f"  ⚠️ Missing subregion")
                                    if meeting_time == 'Unknown':
                                        print(f"  ⚠️ Missing meeting time")
                    
                    # Extract key columns as specified
                    display_cols = ['circle_id', 'region', 'subregion', 'meeting_time', 'member_count', 'new_members', 'max_additions', 'always_hosts', 'sometimes_hosts']
                    
                    # Filter to only include columns that exist in the dataframe
                    if hasattr(circles_df, 'columns'):
                        existing_cols = [col for col in display_cols if col in circles_df.columns]
                    else:
                        existing_cols = []
                    
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
                    
                    # Filter out participants with null Encoded IDs
                    from utils.helpers import get_valid_participants
                    results_df = get_valid_participants(results_df)
                    print(f"🔍 Unmatched table: Using {len(results_df)} valid participants with non-null Encoded IDs")
                    
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
                # Filter out participants with null Encoded IDs for consistent counting
                from utils.helpers import get_valid_participants
                filtered_results = get_valid_participants(st.session_state.results)
                print(f"🔍 CSV Download: Using {len(filtered_results)} valid participants with non-null Encoded IDs")
                
                st.download_button(
                    "Download Results CSV",
                    generate_download_link(filtered_results),
                    "circles_results.csv",
                    "text/csv",
                    key='download-csv'
                )
            
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        if 'debug_mode' in st.session_state.config and st.session_state.config['debug_mode']:
            st.exception(e)

# Define callback for the Match tab
def rebuild_circle_member_lists(circles_df, participants_df):
    """
    Reconstruct complete member lists for all circles directly from participant data.
    This function addresses the disconnect between member_count and actual members lists.
    
    Args:
        circles_df: DataFrame containing circle data
        participants_df: DataFrame containing participant data with circle assignments
        
    Returns:
        Updated DataFrame with rebuilt member lists for each circle
    """
    print("\n🔄 REBUILDING CIRCLE MEMBER LISTS: Starting comprehensive rebuild...")
    
    # Create a deep copy to avoid modifying the original
    updated_circles = circles_df.copy() if isinstance(circles_df, pd.DataFrame) else pd.DataFrame()
    if updated_circles.empty:
        print("⚠️ WARNING: Input circles DataFrame is empty")
    
    # Ensure the participants DataFrame is valid
    if participants_df is None or len(participants_df) == 0:
        print("⚠️ WARNING: No participant data available to rebuild circle member lists")
        return updated_circles
        
    # Find the column that contains circle assignments
    # Use Current_Circle_ID as the primary column (as specified), then fall back to alternatives
    circle_col = None
    preferred_cols = ['Current_Circle_ID', 'assigned_circle', 'circle_id', 'Circle ID', 'proposed_NEW_circles_id']
    
    for col in preferred_cols:
        if col in participants_df.columns:
            circle_col = col
            print(f"✅ Found circle assignment column: '{circle_col}'")
            # Show how many participants have assignments in this column
            assigned_count = participants_df[~participants_df[circle_col].isna()].shape[0]
            total_count = len(participants_df)
            print(f"✅ Found {assigned_count} of {total_count} participants with circle assignments in column '{circle_col}'")
            break
    
    if not circle_col:
        print("⚠️ WARNING: Could not find circle assignment column in participants data")
        print(f"⚠️ Available columns: {participants_df.columns.tolist()}")
        return updated_circles  # No column found to rebuild memberships
    
    # Track which circles were updated and collect statistics
    circles_updated = 0
    all_circle_members = 0
    member_count_fixes = 0
    
    # Step 1: Identify ALL circles, including both original and split circles
    all_circle_ids = set()
    if 'circle_id' in updated_circles.columns:
        all_circle_ids = set(updated_circles['circle_id'].dropna().unique())
        print(f"🔍 Found {len(all_circle_ids)} circles in the input DataFrame")
        
        # Debug: Check if split circles exist in the input DataFrame
        split_circles = [cid for cid in all_circle_ids if 'SPLIT' in cid]
        if split_circles:
            print(f"🔍 Input DataFrame contains {len(split_circles)} split circles: {split_circles[:5]}")
    
    # Add circles from participant data
    participant_circles = set()
    if circle_col in participants_df.columns:
        participant_circles = set(participants_df[circle_col].dropna().unique())
        print(f"🔍 Found {len(participant_circles)} unique circle assignments in participant data")
        
        # Debug: Check if split circles exist in participant data
        split_circles_in_data = [cid for cid in participant_circles if isinstance(cid, str) and 'SPLIT' in cid]
        if split_circles_in_data:
            print(f"🔍 Participant data contains {len(split_circles_in_data)} split circles: {split_circles_in_data[:5]}")
    
    # Find circles that exist in participant data but not in circle DataFrame
    new_circles = participant_circles - all_circle_ids
    if new_circles:
        print(f"⚠️ Found {len(new_circles)} circles in participant data that aren't in circles DataFrame")
        print(f"Sample new circles: {list(new_circles)[:5]}")
    
    # Step 2: Create a mapping of all circle IDs to their members
    circle_member_map = {}
    
    # First, process all participants to map them to circles
    print(f"🔄 Building circle membership map from {len(participants_df)} participants...")
    for _, participant in participants_df.iterrows():
        if circle_col in participant and not pd.isna(participant[circle_col]) and 'Encoded ID' in participant and not pd.isna(participant['Encoded ID']):
            circle_id = str(participant[circle_col])
            member_id = str(participant['Encoded ID'])
            
            if circle_id not in circle_member_map:
                circle_member_map[circle_id] = []
            
            # Add the member to this circle
            if member_id not in circle_member_map[circle_id]:
                circle_member_map[circle_id].append(member_id)
    
    print(f"✅ Built membership map for {len(circle_member_map)} circles")
    
    # Step 3: Update the member lists and counts for all circles in the DataFrame
    # For each circle in the DataFrame
    for idx, circle in updated_circles.iterrows():
        circle_id = circle.get('circle_id', None)
        if not circle_id or pd.isna(circle_id):
            continue
        
        # Convert to string if needed
        circle_id = str(circle_id)
        
        # Get members for this circle from our mapping
        if circle_id in circle_member_map:
            # Members found in participant data
            member_ids = circle_member_map[circle_id]
            
            # Debug for specific test circles
            if circle_id in ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']:
                print(f"🔍 Circle {circle_id}: Found {len(member_ids)} members using column '{circle_col}'")
                if len(member_ids) > 0:
                    print(f"🔍 First few members: {member_ids[:3]}")
                
                # Check for member count mismatch
                if 'member_count' in updated_circles.columns:
                    current_count = updated_circles.at[idx, 'member_count']
                    if current_count != len(member_ids):
                        print(f"⚠️ Member count mismatch for {circle_id}: stored={current_count}, found={len(member_ids)}")
            
            # Update the circle
            updated_circles.at[idx, 'members'] = member_ids
            
            # CRITICAL FIX: Always update member_count to match rebuilt members list
            current_count = updated_circles.at[idx, 'member_count'] if 'member_count' in updated_circles.columns else 0
            if current_count != len(member_ids):
                print(f"🔄 Updating {circle_id} member count: {current_count} → {len(member_ids)}")
                updated_circles.at[idx, 'member_count'] = len(member_ids)
                member_count_fixes += 1
            
            circles_updated += 1
            all_circle_members += len(member_ids)
        else:
            # No members found for this circle - set empty list and zero count
            # This can happen if a circle exists in the circles DataFrame but no participants are assigned to it
            updated_circles.at[idx, 'members'] = []
            
            # Only update member_count if it's not already 0
            if 'member_count' in updated_circles.columns and updated_circles.at[idx, 'member_count'] != 0:
                print(f"⚠️ Circle {circle_id} has no members assigned - setting member_count to 0")
                updated_circles.at[idx, 'member_count'] = 0
                member_count_fixes += 1
    
    # Step 4: Add any circles that exist in participant data but aren't in the DataFrame
    # This ensures we don't miss any circles that exist only in the participant data
    new_circle_rows = []
    for circle_id in new_circles:
        if circle_id in circle_member_map:
            member_ids = circle_member_map[circle_id]
            
            # Debug info
            print(f"🔄 Adding new circle {circle_id} with {len(member_ids)} members")
            
            # Create a new row for this circle with the essential fields
            new_row = {
                'circle_id': circle_id,
                'members': member_ids,
                'member_count': len(member_ids)
            }
            
            # Add to our collection of new rows
            new_circle_rows.append(new_row)
            circles_updated += 1
            all_circle_members += len(member_ids)
    
    # Add all new circles at once if any were found
    if new_circle_rows:
        print(f"🔄 Adding {len(new_circle_rows)} new circles to the DataFrame")
        new_df = pd.DataFrame(new_circle_rows)
        updated_circles = pd.concat([updated_circles, new_df], ignore_index=True)
    
    # Log member count distribution
    if 'member_count' in updated_circles.columns:
        value_counts = updated_circles['member_count'].value_counts().sort_index()
        print(f"\n🔍 MEMBER COUNT DISTRIBUTION:")
        for count, occurrences in value_counts.items():
            print(f"  {count} members: {occurrences} circles")
        
        # Special focus on large circles (those with 11+ members)
        large_circles = updated_circles[updated_circles['member_count'] >= 11]
        if not large_circles.empty:
            print(f"\n🔍 LARGE CIRCLES (11+ members): {len(large_circles)} circles")
            for _, circle in large_circles.iterrows():
                circle_id = circle.get('circle_id', 'Unknown')
                member_count = circle.get('member_count', 0)
                print(f"  {circle_id}: {member_count} members")
    
    # Special case debugging: Check specific circles of interest
    if 'Current_Circle_ID' in participants_df.columns:
        print(f"\n🔍 DETAILED MEMBER ANALYSIS FOR TEST CIRCLES:")
        for test_id in ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']:
            # Find circle in updated DataFrame
            circle_row = updated_circles[updated_circles['circle_id'] == test_id]
            if not circle_row.empty:
                stored_count = circle_row.iloc[0].get('member_count', 0)
                members_list = circle_row.iloc[0].get('members', [])
                members_length = len(members_list) if isinstance(members_list, list) else 0
                
                print(f"\n🔍 DETAILED ANALYSIS OF {test_id}:")
                print(f"  Member count in DataFrame: {stored_count}")
                print(f"  Actual members list length: {members_length}")
                
                # Check if this matches what's in participant data
                if circle_col in participants_df.columns:
                    assigned_members = participants_df[participants_df[circle_col] == test_id]
                    print(f"  Total participants assigned in data: {len(assigned_members)}")
                    
                    # Status breakdown
                    if 'Status' in participants_df.columns:
                        continuing = len(assigned_members[assigned_members['Status'] == 'CONTINUING'])
                        new_members = len(assigned_members[assigned_members['Status'] == 'NEW'])
                        other = len(assigned_members) - continuing - new_members
                        print(f"  Status breakdown: CONTINUING={continuing}, NEW={new_members}, OTHER={other}")
                    
                    # Host status breakdown
                    if 'host_status_standardized' in participants_df.columns:
                        host_breakdown = assigned_members['host_status_standardized'].value_counts()
                        print(f"  Host status breakdown: {dict(host_breakdown)}")
            else:
                print(f"  ⚠️ Test circle {test_id} not found in updated circles DataFrame")
    
    print(f"✅ Successfully rebuilt member lists for {circles_updated} circles with a total of {all_circle_members} members")
    print(f"✅ Fixed {member_count_fixes} member count inconsistencies")
    
    return updated_circles

def test_circle_splitting():
    """Test function to directly test the circle splitting functionality"""
    st.info("Running direct test of circle splitting functionality...")
    
    if 'matched_circles' not in st.session_state or st.session_state.matched_circles is None:
        st.error("No data available. Please upload a CSV file and run the matching algorithm first.")
        return
    
    try:
        # Get the matched circles from session state
        circles_data = st.session_state.matched_circles
        participants_data = st.session_state.processed_data
        
        # Debug log what we found
        st.write(f"Found {len(circles_data)} circles in session state")
        st.write(f"Found {len(participants_data)} participants in session state")
        
        # Show available columns in participants data - to see how circles are assigned
        st.write("Available columns in participants data:")
        st.write(participants_data.columns.tolist())
        
        # Attempt to identify which column contains circle assignments
        circle_col = None
        # First, check if we have Current_Circle_ID as specified
        preferred_cols = ['Current_Circle_ID', 'assigned_circle', 'circle_id', 'Circle ID', 'proposed_NEW_circles_id']
        
        for col in preferred_cols:
            if col in participants_data.columns:
                circle_col = col
                st.write(f"Found circle assignment column: '{circle_col}'")
                # Show how many participants have circle assignments
                assigned_count = participants_data[~participants_data[circle_col].isna()].shape[0]
                st.write(f"Participants with circle assignments: {assigned_count} of {len(participants_data)}")
                # Show sample of different circle IDs in use
                sample_circles = participants_data[circle_col].dropna().unique()[:5]
                st.write(f"Sample circle IDs: {sample_circles.tolist()}")
                
                # For Current_Circle_ID specifically, also check status distribution
                if col == 'Current_Circle_ID':
                    st.write("Distribution of participants with Current_Circle_ID by Status:")
                    if 'Status' in participants_data.columns:
                        status_counts = participants_data[~participants_data['Current_Circle_ID'].isna()]['Status'].value_counts()
                        st.write(status_counts)
                break
        
        if not circle_col:
            st.warning("Could not identify which column contains circle assignments")
            # Show all columns to help identify the right one
            st.write("Please review available columns to identify circle assignments:")
            for col in participants_data.columns:
                unique_values = participants_data[col].dropna().unique()
                if len(unique_values) < 200:  # Only show if it has a reasonable number of values
                    st.write(f"- {col}: {unique_values[:5]}")
        
        # Import circle splitting functionality
        from modules.circle_splitter import split_large_circles
        
        # Rebuild circle member lists from participant data - NEW APPROACH
        st.subheader("Rebuilding Circle Member Lists")
        rebuilt_circles = rebuild_circle_member_lists(circles_data, participants_data)
        
        # Create test circles from the rebuilt data
        test_circles = []
        test_participants = participants_data.copy()
        
        # Import circle metadata manager for consistent member access (for comparison only)
        from utils.circle_metadata_manager import CircleMetadataManager
        
        # Create a new metadata manager or get it from session state if it exists
        if 'circle_metadata_manager' in st.session_state:
            metadata_manager = st.session_state.circle_metadata_manager
            st.write("Using existing CircleMetadataManager from session state")
        else:
            metadata_manager = CircleMetadataManager()
            # Initialize from current data
            metadata_manager.initialize_from_optimizer(circles_data.to_dict('records'), participants_data)
            st.write("Created new CircleMetadataManager")
            
        # ENHANCED: Find all large circles (11+ members) first
        st.subheader("Identifying Large Circles")
        
        # Check for large circles in the rebuilt data
        large_circles = rebuilt_circles[rebuilt_circles['member_count'] >= 11]
        if not large_circles.empty:
            st.write(f"Found {len(large_circles)} large circles (11+ members) in the data")
            st.dataframe(large_circles[['circle_id', 'member_count']])
            
            # Add all large circles to our test set automatically
            for _, circle in large_circles.iterrows():
                test_circles.append(circle.to_dict())
                st.write(f"Added large circle {circle['circle_id']} with {circle['member_count']} members to test set")
        else:
            st.warning("No large circles (11+ members) found in the data")
        
        # For extra certainty, also check our priority test circles
        priority_test_ids = ['IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']
        st.write(f"Looking for specific priority test circles: {', '.join(priority_test_ids)}")
        
        # ENHANCED: Direct participant counting to ensure accurate counts for test circles
        st.subheader("Direct Participant Analysis")
        
        # Find the correct column for circle assignments
        if circle_col:
            for circle_id in priority_test_ids:
                # Count participants directly
                participants_in_circle = participants_data[participants_data[circle_col] == circle_id]
                direct_member_count = len(participants_in_circle)
                
                st.write(f"Direct member count for {circle_id}: {direct_member_count}")
                
                if direct_member_count >= 11:
                    st.write(f"⚠️ {circle_id} has {direct_member_count} members but may not be properly identified")
                    # Force this circle to be included regardless of what's in the DataFrame
                    
                    # Check if it's already in our test set
                    already_added = any(c.get('circle_id') == circle_id for c in test_circles)
                    
                    if not already_added:
                        # Extract member IDs
                        member_ids = []
                        if 'Encoded ID' in participants_in_circle.columns:
                            member_ids = [str(id) for id in participants_in_circle['Encoded ID'].tolist() if id is not None and not pd.isna(id)]
                        
                        # Create a synthetic circle with the direct member count
                        synthetic_circle = {
                            'circle_id': circle_id,
                            'member_count': direct_member_count,
                            'members': member_ids,
                            'always_hosts': 1,  # Assume at least 1 always host
                            'sometimes_hosts': 1  # Assume at least 1 sometimes host
                        }
                        
                        # Add to our test set
                        test_circles.append(synthetic_circle)
                        st.write(f"Manually added {circle_id} with direct count of {direct_member_count} members")
        
        # Now check our test circles against the rebuilt data (still useful for debugging)
        st.subheader("Detailed Circle Validation")
        
        for circle_id in priority_test_ids:
            # Look for the circle in our rebuilt data
            matches = rebuilt_circles[rebuilt_circles['circle_id'] == circle_id]
            if not matches.empty:
                test_circle = matches.iloc[0].to_dict()
                
                # Get rebuilt members
                members_from_rebuild = test_circle.get('members', [])
                if isinstance(members_from_rebuild, list):
                    members_length = len(members_from_rebuild)
                else:
                    members_length = 0
                    st.warning(f"Unexpected type for members_from_rebuild: {type(members_from_rebuild)}")
                
                # Try to get members using different methods, with error handling
                try:
                    # Method 1: Using metadata manager (may fail if circle not in manager)
                    members_from_manager = metadata_manager.get_circle_members(circle_id)
                    manager_length = len(members_from_manager)
                except Exception as e:
                    print(f"⚠️ Error getting members from metadata manager: {str(e)}")
                    members_from_manager = []
                    manager_length = 0
                
                try:
                    # Method 2: Direct normalization from the original circle data
                    from utils.data_standardization import normalize_member_list
                    circle_matches = circles_data[circles_data['circle_id'] == circle_id]
                    if not circle_matches.empty:
                        original_circle = circle_matches.iloc[0].to_dict()
                        members_from_normalization = normalize_member_list(original_circle.get('members', []))
                        norm_length = len(members_from_normalization)
                    else:
                        members_from_normalization = []
                        norm_length = 0
                except Exception as e:
                    print(f"⚠️ Error normalizing members: {str(e)}")
                    members_from_normalization = []
                    norm_length = 0
                
                # Log what we found with each method
                st.write(f"Found circle {circle_id} in DataFrame with {test_circle.get('member_count', 0)} members")
                st.write(f"- Method 1 (MetadataManager): {manager_length} members")
                st.write(f"- Method 2 (Rebuilt): {members_length} members")
                st.write(f"- Method 3 (Normalization): {norm_length} members")
                
                # Compare against direct participant count
                if circle_col:
                    direct_count = len(participants_data[participants_data[circle_col] == circle_id])
                    st.write(f"- Method 4 (Direct participant count): {direct_count} members")
                    
                    if direct_count != members_length:
                        st.warning(f"⚠️ Member count mismatch: {members_length} (rebuilt) vs {direct_count} (direct count)")
                
                # Analyze differences
                if len(members_from_manager) > 0 and members_length > 0:
                    # Show differences between member lists
                    manager_set = set(members_from_manager)
                    rebuild_set = set(members_from_rebuild) if isinstance(members_from_rebuild, list) else set()
                    
                    # Members in manager but not in rebuild
                    missing_from_rebuild = manager_set - rebuild_set
                    if missing_from_rebuild:
                        st.write(f"Members in manager but not in rebuild: {missing_from_rebuild}")
                    
                    # Members in rebuild but not in manager
                    missing_from_manager = rebuild_set - manager_set
                    if missing_from_manager:
                        st.write(f"Members in rebuild but not in manager: {missing_from_manager}")
                
                # Check if this circle is already in our test set
                already_added = any(c.get('circle_id') == circle_id for c in test_circles)
                
                if not already_added:
                    # Add to test circles if not already added
                    test_circles.append(test_circle)
                    st.write(f"Added {circle_id} to test circles from rebuild data: {members_length} members")
            else:
                st.warning(f"Test circle {circle_id} not found in rebuilt circles DataFrame")
        
        # If no test circles found, create synthetic ones with a clear note
        if not test_circles:
            st.warning("No test circles found in data. Creating a synthetic test circle.")
            # Create a synthetic circle with 11+ members
            member_ids = test_participants['Encoded ID'].iloc[:12].tolist()
            test_circles.append({
                'circle_id': 'IP-TST-01',
                'members': member_ids,
                'member_count': len(member_ids),
                'always_hosts': 1,
                'sometimes_hosts': 2
            })
        
        # Convert to DataFrame for processing
        test_circles_df = pd.DataFrame(test_circles)
        
        # Run the circle splitting function directly
        st.write("Running circle splitting function...")
        print("🔴 TEST: Running direct circle splitting test")
        
        # Import the circle splitting function
        from modules.circle_splitter import split_large_circles
        
        updated_circles, split_summary = split_large_circles(test_circles_df, test_participants)
        
        # Store the split summary in session state for UI components to use
        st.session_state.split_circle_summary = split_summary
        
        # Display results
        st.subheader("Split Circle Results")
        st.write(f"Original circles: {len(test_circles)}")
        st.write(f"Total circles examined: {split_summary['total_circles_examined']}")
        st.write(f"Large circles found: {split_summary['total_large_circles_found']}")
        st.write(f"Circles successfully split: {split_summary['total_circles_successfully_split']}")
        st.write(f"New circles created: {split_summary['total_new_circles_created']}")
        
        # Display details of each split
        if split_summary['split_details']:
            st.subheader("Split Details")
            for detail in split_summary['split_details']:
                st.write(f"Original circle: {detail['original_circle_id']}")
                st.write(f"Member count: {detail['member_count']}")
                st.write(f"Split into {len(detail['new_circle_ids'])} circles:")
                for i, new_id in enumerate(detail['new_circle_ids']):
                    member_count = detail['member_counts'][i] if i < len(detail['member_counts']) else "?"
                    st.write(f"  - {new_id} with {member_count} members")
                    # Show host distribution if available
                    if 'always_hosts' in detail and 'sometimes_hosts' in detail:
                        always = detail['always_hosts'][i] if i < len(detail['always_hosts']) else 0
                        sometimes = detail['sometimes_hosts'][i] if i < len(detail['sometimes_hosts']) else 0
                        st.write(f"    Always Hosts: {always}, Sometimes Hosts: {sometimes}")
                st.write("---")
        else:
            st.warning("No circles were split.")
            
        # Display circles that couldn't be split
        if split_summary['circles_unable_to_split']:
            st.subheader("Circles Unable to Split")
            for circle in split_summary['circles_unable_to_split']:
                st.write(f"Circle {circle['circle_id']} ({circle['member_count']} members)")
                st.write(f"Reason: {circle['reason']}")
                st.write("---")
                
        # Display the updated circles DataFrame (check if DataFrame or list)
        if isinstance(updated_circles, pd.DataFrame) and not updated_circles.empty:
            st.subheader("Updated Circles")
            st.dataframe(updated_circles)
            
            # Count split circles
            split_circles = updated_circles[updated_circles['circle_id'].str.contains('SPLIT')]
            st.write(f"Found {len(split_circles)} split circles in the updated dataset")
        elif isinstance(updated_circles, list) and len(updated_circles) > 0:
            st.subheader("Updated Circles")
            # Convert list to DataFrame for display
            updated_df = pd.DataFrame(updated_circles)
            st.dataframe(updated_df)
            
            # Count split circles
            split_circle_count = sum(1 for c in updated_circles if 'SPLIT' in c.get('circle_id', ''))
            st.write(f"Found {split_circle_count} split circles in the updated dataset")
        else:
            st.warning("No updated circles returned.")
            
        # Use our standardized render function to display the summary
        st.subheader("Standardized Circle Split Summary")
        from modules.ui_components import render_split_circle_summary
        render_split_circle_summary()
    
    except Exception as e:
        st.error(f"Error during circle splitting test: {str(e)}")
        st.write("Exception details:")
        import traceback
        st.code(traceback.format_exc())
        st.code(traceback.format_exc())
        print("🔴 CRITICAL ERROR IN CIRCLE SPLITTING TEST:")
        print(traceback.format_exc())

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
