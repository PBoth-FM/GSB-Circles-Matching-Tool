import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import os
from modules.data_loader import load_data, validate_data
from modules.data_processor import process_data, normalize_data
from modules.optimizer import run_matching_algorithm
from modules.ui_components import (
    render_match_tab, 
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

def render_documentation_tab():
    """Render the documentation tab content"""
    st.subheader("Documentation")

    # Read and display the documentation file
    try:
        with open('documentation.md', 'r') as f:
            documentation_content = f.read()

        # Display the markdown content
        st.markdown(documentation_content)

    except FileNotFoundError:
        st.error("Documentation file not found. Please ensure documentation.md exists in the project root.")
    except Exception as e:
        st.error(f"Error loading documentation: {str(e)}")

def main():
    st.title("GSB Circles Matching Tool")

    # Create tabs for navigation, moved Demographics after Match per user request
    # Added Documentation tab between Demographics and Debug
    tab1, tab2, tab3, tab4 = st.tabs(["Match", "Demographics", "Documentation", "Debug"])

    with tab1:
        # Use our custom match tab function instead of the imported one
        match_tab_callback()

    with tab2:
        render_demographics_tab()

    with tab3:
        render_documentation_tab()

    with tab4:
        render_debug_tab()

def run_optimization():
    """Run the optimization algorithm and store results in session state"""

    # 🚀 CRITICAL DEBUG: App optimization entry point
    print(f"\n🚀🚀🚀 RUN_OPTIMIZATION CALLED IN APP.PY! 🚀🚀🚀")

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

    # Set fixed optimization weights (no UI configuration needed)
    st.session_state.config['optimization_weight_location'] = 3.0
    st.session_state.config['optimization_weight_time'] = 2.0

    try:
        with st.spinner("Running matching algorithm..."):
            start_time = time.time()

            # Use the original data without any test participants
            # Run the matching algorithm with enhanced return values for debugging

            # 🚀 CRITICAL DEBUG: About to call main algorithm
            print(f"\n🚀 ABOUT TO CALL run_matching_algorithm!")
            print(f"  Data shape: {st.session_state.processed_data.shape}")
            print(f"  Config: {st.session_state.config}")

            from modules.optimizer import run_matching_algorithm
            print(f"  ✅ Successfully imported run_matching_algorithm")

            results, matched_circles, unmatched_participants = run_matching_algorithm(
                st.session_state.processed_data,
                st.session_state.config
            )

            print(f"  ✅ run_matching_algorithm completed successfully!")

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

            # Apply co-leader assignment logic before storing results
            print("\n🎯 APPLYING CO-LEADER ASSIGNMENT LOGIC")
            try:
                from modules.co_leader_assignment import assign_co_leaders, validate_co_leader_assignments

                # Apply co-leader assignment to results
                results_with_co_leaders = assign_co_leaders(results, debug_mode=st.session_state.config.get('debug_mode', False))

                # Validate the assignments
                validation = validate_co_leader_assignments(results_with_co_leaders)

                if validation['valid']:
                    print("  ✅ Co-leader assignments completed successfully")
                    print(f"  📊 Statistics: {validation['statistics']}")
                    results = results_with_co_leaders
                else:
                    print("  ⚠️ Co-leader assignment validation issues:")
                    for issue in validation['issues']:
                        print(f"    - {issue}")
                    # Still use the results with co-leader assignments, but log the issues
                    results = results_with_co_leaders

            except Exception as e:
                print(f"  ⚠️ Error in co-leader assignment: {str(e)}")
                print("  Continuing with original results without co-leader assignments")

            # Validate same-person constraint (prevent participants with same base ID in same circle)
            print("\n🔒 VALIDATING SAME-PERSON CONSTRAINT")
            try:
                from modules.same_person_constraint_test import validate_same_person_constraint

                validation_result = validate_same_person_constraint(results)
                print(f"  {validation_result['message']}")

                if not validation_result['valid']:
                    print("  ⚠️ Same-person constraint violations found:")
                    for violation in validation_result['violations']:
                        print(f"    Circle {violation['circle_id']}: Base ID {violation['base_encoded_id']} appears {violation['count']} times")
                        print(f"      Participants: {violation['duplicate_participants']}")

                    # Store validation results for display in UI
                    if 'same_person_violations' not in st.session_state:
                        st.session_state.same_person_violations = []
                    st.session_state.same_person_violations = validation_result['violations']
                else:
                    # Clear any previous violations
                    if 'same_person_violations' in st.session_state:
                        st.session_state.same_person_violations = []

            except Exception as e:
                print(f"  ⚠️ Error validating same-person constraint: {str(e)}")

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
        # CRITICAL FIX: Complete session state reset when new data is uploaded
        print("\n🔄 COMPLETE SESSION STATE RESET - New data uploaded")

        # Clear ALL matching-related session state variables
        session_keys_to_clear = [
            'results', 'matched_circles', 'unmatched_participants',
            'circle_manager', 'optimization_results', 'circle_eligibility_logs',
            'match_statistics', 'total_diversity_score', 'seattle_debug_logs',
            'processed_data', 'df', 'validation_errors', 'deduplication_messages'
        ]

        for key in session_keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
                print(f"  ✅ Cleared session state: {key}")

        # Force clear any cached values
        st.cache_data.clear()
        print("  ✅ Cleared all cached data")

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

            # Initialize max_circle_size if not set
            if 'max_circle_size' not in st.session_state:
                st.session_state.max_circle_size = 8

            # Maximum Circle Size configuration
            st.session_state.max_circle_size = st.number_input(
                "Maximum Circle Size",
                min_value=5,
                max_value=10,
                value=st.session_state.max_circle_size,
                help="Note: this will not affect continuing participants"
            )

            # Only show Debug Mode as a configurable option
            st.session_state.config['debug_mode'] = st.checkbox(
                "Debug Mode", 
                value=st.session_state.config['debug_mode'],
                help="Enable to see detailed logs and diagnostic information"
            )

            # Add run button once data is loaded
            if st.button("Run Matching Algorithm", key="match_run_algorithm_button"):
                run_optimization()

            # Check for session state integrity and restore if needed
            if ('results' not in st.session_state or st.session_state.results is None) and 'backup_results' in st.session_state:
                st.session_state.results = st.session_state.backup_results
                st.session_state.matched_circles = st.session_state.get('backup_matched_circles', None)
                st.info("🔄 Restored results data from backup")

            # Display results if available
            if hasattr(st.session_state, 'results') and st.session_state.results is not None:
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
                        # Count circles from the same data source as Circle Composition table
                        if 'results' in st.session_state and st.session_state.results is not None:
                            # Get unique circle IDs from results, excluding UNMATCHED
                            results_for_circles = st.session_state.results.copy()
                            if 'proposed_NEW_circles_id' in results_for_circles.columns:
                                # Count all unique circles (including zero-member circles)
                                all_circle_ids = results_for_circles['proposed_NEW_circles_id'].dropna()
                                unique_circles = all_circle_ids[all_circle_ids != 'UNMATCHED'].unique()
                                circles_count = len(unique_circles)
                                st.metric("Circles Created", circles_count)
                            else:
                                st.metric("Circles Created", 0)
                        else:
                            st.metric("Circles Created", 0)

                    with col2:
                        # Use the standardized matched count
                        st.metric("Participants Matched", match_stats['matched_participants'])

                    with col3:
                        # Use the standardized match rate
                        st.metric("Match Success Rate", f"{match_stats['match_rate']:.1f}%")

                # Create size histogram using CSV data - moved after CSV table generation
                st.subheader("Circle Size Distribution")

                # Use the CSV circles data that was just generated above
                if 'results' in st.session_state and st.session_state.results is not None:
                    results_df = st.session_state.results.copy()

                    # Check if we have the required columns
                    required_cols = ['proposed_NEW_circles_id']

                    if all(col in results_df.columns for col in required_cols):
                        # Filter out unmatched participants
                        matched_results = results_df[
                            (results_df['proposed_NEW_circles_id'].notna()) & 
                            (results_df['proposed_NEW_circles_id'] != 'UNMATCHED')
                        ].copy()

                        if len(matched_results) > 0:
                            # Group by circle ID and count members
                            circle_sizes = matched_results.groupby('proposed_NEW_circles_id').size()

                            # Filter out zero-member circles (though there shouldn't be any)
                            circle_sizes = circle_sizes[circle_sizes > 0]

                            if len(circle_sizes) > 0:
                                # Calculate size distribution
                                size_counts = circle_sizes.value_counts().sort_index()

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
                                avg_size = circle_sizes.mean()
                                median_size = circle_sizes.median()

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Average Circle Size", f"{avg_size:.1f}")
                                with col2:
                                    st.metric("Median Circle Size", f"{median_size:.0f}")
                            else:
                                st.warning("No circles with members found.")
                        else:
                            st.warning("No matched participants found.")
                    else:
                        st.warning("Required columns not available for circle size analysis.")
                else:
                    st.warning("No results data available for circle size analysis.")

                # Same-Person Constraint Validation
                st.subheader("Same-Person Constraint Validation")
                from modules.ui_components import render_same_person_constraint_validation
                render_same_person_constraint_validation()

                # Circle Composition from CSV - Direct from Results Data
                st.subheader("Circle Composition from CSV")

                if 'results' in st.session_state and st.session_state.results is not None:
                    results_df = st.session_state.results.copy()

                    # Check if we have the required columns
                    required_cols = ['proposed_NEW_circles_id', 'Derived_Region', 'proposed_NEW_Subregion', 
                                   'proposed_NEW_DayTime', 'Encoded ID', 'Status', 'co_leader_max_new_members', 
                                   'host_status_standardized']

                    missing_cols = [col for col in required_cols if col not in results_df.columns]

                    if missing_cols:
                        st.warning(f"Missing required columns for CSV table: {missing_cols}")
                    else:
                        # Filter out unmatched participants
                        matched_results = results_df[
                            (results_df['proposed_NEW_circles_id'].notna()) & 
                            (results_df['proposed_NEW_circles_id'] != 'UNMATCHED')
                        ].copy()

                        if len(matched_results) > 0:
                            # Group by circle ID and aggregate data
                            circle_groups = matched_results.groupby('proposed_NEW_circles_id')

                            csv_circles_data = []

                            for circle_id, group in circle_groups:
                                # Calculate aggregated values
                                member_count = len(group)
                                new_members = len(group[group['Status'] == 'NEW'])

                                # Get first values for region, subregion, meeting time (should be same for all members)
                                region = group['Derived_Region'].iloc[0] if not group['Derived_Region'].isna().all() else 'Unknown'
                                subregion = group['proposed_NEW_Subregion'].iloc[0] if not group['proposed_NEW_Subregion'].isna().all() else 'Unknown'
                                meeting_time = group['proposed_NEW_DayTime'].iloc[0] if not group['proposed_NEW_DayTime'].isna().all() else 'Unknown'

                                # SOLUTION 1: Use stored optimization results with new circle logic
                                # Check if this is a new circle (contains "-NEW-" in the ID)
                                is_new_circle = "-NEW-" in circle_id

                                if is_new_circle:
                                    # New circles always have max_additions = configured maximum circle size
                                    max_additions = st.session_state.get('max_circle_size', 8)
                                    print(f"✅ New circle {circle_id}: Setting max_additions={max_additions} (maximum capacity)")
                                else:
                                    # Continuing circle: use stored optimization results
                                    max_additions = 0  # Default fallback

                                    # Try to get from CircleMetadataManager
                                    if 'circle_metadata_manager' in st.session_state and st.session_state.circle_metadata_manager:
                                        manager = st.session_state.circle_metadata_manager
                                        if hasattr(manager, 'circles') and circle_id in manager.circles:
                                            stored_max_additions = manager.circles[circle_id].get('max_additions', None)
                                            if stored_max_additions is not None:
                                                max_additions = stored_max_additions
                                                print(f"✅ Continuing circle {circle_id}: Using stored max_additions={max_additions}")
                                            else:
                                                print(f"⚠️ No max_additions found in manager for continuing circle {circle_id}")
                                        else:
                                            print(f"⚠️ Continuing circle {circle_id} not found in metadata manager")

                                    # If we still don't have a value, try to get from matched_circles
                                    if max_additions == 0 and 'matched_circles' in st.session_state:
                                        matched_circles = st.session_state.matched_circles
                                        if isinstance(matched_circles, pd.DataFrame) and not matched_circles.empty:
                                            # Look for this circle in the matched_circles DataFrame
                                            circle_row = matched_circles[matched_circles['circle_id'] == circle_id]
                                            if not circle_row.empty and 'max_additions' in matched_circles.columns:
                                                stored_max_additions = circle_row['max_additions'].iloc[0]
                                                if pd.notna(stored_max_additions):
                                                    max_additions = int(stored_max_additions)
                                                    print(f"✅ Continuing circle {circle_id}: Using max_additions={max_additions} from matched_circles")

                                    # Final fallback for continuing circles: if we still have 0 but there are new members assigned, 
                                    # set max_additions to at least match the new members count
                                    if max_additions == 0 and new_members > 0:
                                        max_additions = new_members
                                        print(f"⚠️ Continuing circle {circle_id}: Fallback max_additions={max_additions} to match new_members")

                                # Count host status
                                always_hosts = len(group[group['host_status_standardized'] == 'ALWAYS'])
                                sometimes_hosts = len(group[group['host_status_standardized'] == 'SOMETIMES'])

                                csv_circles_data.append({
                                    'Circle Id': circle_id,
                                    'Region': region,
                                    'Subregion': subregion,
                                    'Meeting Time': meeting_time,
                                    'Member Count': member_count,
                                    'New Members': new_members,
                                    'Max Additions': max_additions,
                                    'Always Hosts': always_hosts,
                                    'Sometimes Hosts': sometimes_hosts
                                })

                            # Create DataFrame and display
                            if csv_circles_data:
                                csv_circles_df = pd.DataFrame(csv_circles_data)
                                csv_circles_df = csv_circles_df.sort_values('Circle Id')
                                st.dataframe(csv_circles_df)


                            else:
                                st.warning("No circle data could be generated from CSV results.")
                        else:
                            st.warning("No matched participants found in results data.")
                else:
                    st.warning("No results data available. Please run the matching algorithm first.")

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

                # Before generating the download link, back up the results data
                st.session_state.backup_results = st.session_state.results
                st.session_state.backup_matched_circles = st.session_state.get('matched_circles', None)

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