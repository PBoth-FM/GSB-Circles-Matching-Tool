import time
import sys
import json
import os
import pandas as pd
import pulp
import streamlit as st
from utils.helpers import generate_circle_id, determine_unmatched_reason
from modules.data_processor import is_time_compatible
from utils.region_mapper import (
    normalize_region_name,
    extract_region_code_from_circle_id,
    get_region_from_circle_or_participant,
    map_circles_to_regions
)

# Global debug flag to trace region mapping issues
TRACE_REGION_MAPPING = True

# REMOVED: Houston debug logs - focusing only on Seattle region for the matching optimization
# houston_debug_logs = []

# Initialize debug counter for tracking
DEBUG_ELIGIBILITY_COUNTER = 0

# File path for circle eligibility logs
CIRCLE_ELIGIBILITY_LOGS_PATH = './debug_data/circle_eligibility_logs.json'

# Initialize directory for debug data
os.makedirs(os.path.dirname(CIRCLE_ELIGIBILITY_LOGS_PATH), exist_ok=True)

# REFACTORED APPROACH: We completely removed the global circle_eligibility_logs variable
# Instead, we create the dictionary in optimize_region_v2 
# and pass it explicitly as parameters to functions that need it

# Create a special log for tracking the circle eligibility logs
def debug_eligibility_logs(message):
    """Helper function to print standardized circle eligibility debug logs"""
    print(f"🔍 CIRCLE ELIGIBILITY DEBUG: {message}")

# Function to save circle eligibility logs to file
def save_circle_eligibility_logs_to_file(logs, region="all"):
    """
    Save circle eligibility logs to a JSON file.
    Merges with existing logs if file exists.

    Args:
        logs: Dictionary of circle eligibility logs
        region: Region being processed (for debugging)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # CRITICAL DEBUG: Print what we're actually getting
        print(f"\n🚨 CRITICAL LOGS DEBUG for {region}:")
        print(f"🚨 Received {len(logs)} logs to save")
        if logs:
            sample_keys = list(logs.keys())[:3]
            print(f"🚨 Sample keys: {sample_keys}")
            for key in sample_keys:
                print(f"🚨 Sample entry for {key}: {logs[key].get('is_eligible', 'unknown')}")

        # First check if the file exists and load existing logs
        existing_logs = {}
        if os.path.exists(CIRCLE_ELIGIBILITY_LOGS_PATH):
            try:
                with open(CIRCLE_ELIGIBILITY_LOGS_PATH, 'r') as f:
                    data = json.load(f)
                    if "logs" in data and isinstance(data["logs"], dict):
                        existing_logs = data["logs"]
                        print(f"📂 Loaded {len(existing_logs)} existing logs from file")

                        # Debug what regions we have in existing logs
                        existing_regions = set(v.get('region', 'unknown') for v in existing_logs.values())
                        print(f"📂 Existing logs contain these regions: {existing_regions}")
                    else:
                        print(f"⚠️ WARNING: File exists but doesn't contain valid 'logs' entry")
            except Exception as e:
                print(f"⚠️ Could not load existing logs from file: {str(e)}")
        else:
            print(f"📂 No existing logs file found at {CIRCLE_ELIGIBILITY_LOGS_PATH}")

        # Verify input logs before merging
        if not isinstance(logs, dict):
            print(f"⚠️ WARNING: Input logs is not a dictionary! Type: {type(logs)}")
            print(f"Converting to empty dict to avoid errors")
            logs = {}

        # ENHANCED MERGE: Make explicit deep copy of both dictionaries to prevent reference issues
        merged_logs = {}

        # First copy existing logs
        for key, value in existing_logs.items():
            if isinstance(value, dict):
                merged_logs[key] = value.copy()  # Deep copy for dictionaries
            else:
                merged_logs[key] = value  # Direct assignment for non-dict values

        # Then merge in new logs (they take precedence)
        for key, value in logs.items():
            if isinstance(value, dict):
                merged_logs[key] = value.copy()  # Deep copy for dictionaries
            else:
                merged_logs[key] = value  # Direct assignment for non-dict values

        # Add timestamp and region info
        data_to_save = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "region": region,
                "total_logs": len(merged_logs),
                "added_logs": len(logs),
                "existing_logs": len(existing_logs),
                "regions_processed": region if region == "all" else [region]
            },
            "logs": merged_logs
        }

        print(f"\n📝 SAVING CIRCLE ELIGIBILITY LOGS: {len(logs)} new entries + {len(existing_logs)} existing = {len(merged_logs)} total")

        # Write to file
        with open(CIRCLE_ELIGIBILITY_LOGS_PATH, 'w') as f:
            json.dump(data_to_save, f, indent=2)

        # CRITICAL VERIFICATION: Read the file back to ensure it was saved correctly
        try:
            with open(CIRCLE_ELIGIBILITY_LOGS_PATH, 'r') as f:
                verification_data = json.load(f)

            verification_logs = verification_data.get("logs", {})
            print(f"✅ VERIFICATION: Successfully read back {len(verification_logs)} logs from file")

            # Verify key counts match
            if len(verification_logs) != len(merged_logs):
                print(f"⚠️ WARNING: Verification logs count ({len(verification_logs)}) doesn't match merged logs count ({len(merged_logs)})")
        except Exception as e:
            print(f"⚠️ WARNING: Verification read failed: {str(e)}")

        print(f"✅ Successfully saved merged logs to {CIRCLE_ELIGIBILITY_LOGS_PATH}")
        return True

    except Exception as e:
        print(f"❌ ERROR saving circle eligibility logs to file: {str(e)}")
        return False

# Function to load circle eligibility logs from file
def load_circle_eligibility_logs_from_file():
    """
    Load circle eligibility logs from JSON file.

    Returns:
        dict: Circle eligibility logs or empty dict if file doesn't exist
    """
    try:
        # CRITICAL FIX: Add debug flag to check if we should force recalculation
        # This ensures we're using the latest time compatibility logic
        force_recalculate = True

        if force_recalculate:
            print("🔄 COMPATIBILITY FIX: Forcing recalculation of circle eligibility instead of loading from file")
            print("  This ensures the latest time compatibility enhancements are used")
            return {}

        if os.path.exists(CIRCLE_ELIGIBILITY_LOGS_PATH):
            with open(CIRCLE_ELIGIBILITY_LOGS_PATH, 'r') as f:
                data = json.load(f)

            if "logs" in data and isinstance(data["logs"], dict):
                print(f"📂 Loaded {len(data['logs'])} circle eligibility logs from file")
                print(f"📄 File timestamp: {data.get('metadata', {}).get('timestamp', 'unknown')}")
                return data["logs"]
            else:
                print("⚠️ Invalid format in circle eligibility logs file")
                return {}
        else:
            print("ℹ️ No circle eligibility logs file found")
            return {}

    except Exception as e:
        print(f"❌ ERROR loading circle eligibility logs from file: {str(e)}")
        return {}

# Function to directly update circle eligibility logs in session state
def update_session_state_eligibility_logs(circle_logs=None):
    """
    Helper function to ensure circle eligibility logs are properly stored in session state.
    Call this whenever eligibility logs are updated to ensure consistency.

    Args:
        circle_logs: Dictionary of circle eligibility logs to store in session state
                    If None, an empty dictionary will be used
    """
    try:
        import streamlit as st

        # Use provided logs or create an empty dictionary
        logs_to_store = circle_logs if circle_logs is not None else {}

        # ENHANCED DIAGNOSTICS - Print detailed information
        print(f"\n🔍 CRITICAL DIAGNOSTIC: update_session_state_eligibility_logs called")
        print(f"🔍 Received logs dictionary with {len(logs_to_store)} entries")

        # Debug information about the provided dictionary
        print(f"🔍 Type of logs_to_store: {type(logs_to_store)}")
        print(f"🔍 Is logs_to_store a dictionary? {isinstance(logs_to_store, dict)}")

        # Show the first few circle IDs if any exist
        if logs_to_store and isinstance(logs_to_store, dict):
            circle_ids = list(logs_to_store.keys())
            print(f"🔍 Circle IDs in logs: {circle_ids[:5]}{'...' if len(circle_ids) > 5 else ''}")
            # Show details of first log entry for debugging
            first_id = circle_ids[0] if circle_ids else None
            if first_id:
                print(f"🔍 Sample log entry for {first_id}: {logs_to_store[first_id]}")
        else:
            print("❌ CRITICAL ERROR: Provided logs dictionary is empty or not a dictionary!")
            # If it's not a dictionary, initialize it as one
            if not isinstance(logs_to_store, dict):
                print("🔧 FIXING: Initializing logs_to_store as a dictionary")
                logs_to_store = {}

        # Create session state if needed
        if 'circle_eligibility_logs' not in st.session_state:
            try:
                st.session_state.circle_eligibility_logs = {}
                print("🔍 Created new circle_eligibility_logs in session state")
            except Exception as e:
                print(f"❌ ERROR creating session state: {str(e)}")
                return False

        # Count before update for debugging
        try:
            before_count = len(st.session_state.circle_eligibility_logs)

            # Make a deep copy of the logs to prevent reference issues
            logs_copy = {}
            for key, value in logs_to_store.items():
                if isinstance(value, dict):
                    logs_copy[key] = value.copy()  # Copy each inner dictionary
                else:
                    logs_copy[key] = value

            # Copy logs to session state
            st.session_state.circle_eligibility_logs.update(logs_copy)

            # Count after update for debugging
            after_count = len(st.session_state.circle_eligibility_logs)
            print(f"🔍 Updated session state: {before_count} → {after_count} logs")

            # Additional verification
            if after_count == 0:
                print("❌ CRITICAL ERROR: Session state circle_eligibility_logs is still empty after update!")
                return False

            # We no longer need the global flag since we're using parameter passing
            # This implementation relies on the session state to track logs directly
            print(f"🔍 Successfully updated session state with circle eligibility logs")

            # For additional safety, verify a sample entry was properly copied
            if after_count > 0:
                sample_key = next(iter(st.session_state.circle_eligibility_logs))
                print(f"✅ Verification: Session state contains entry for {sample_key}")

            return True
        except Exception as e:
            print(f"❌ ERROR updating session state: {str(e)}")
            return False
    except Exception as e:
        print(f"❌ CRITICAL ERROR in update_session_state_eligibility_logs: {str(e)}")
        return False

# Add a direct check to display eligibility logs at import time
debug_eligibility_logs(f"Module initialized with transition to parameter-based circle eligibility logs")

# Example participants and circles for testing (removed Houston test case)
test_participants = ['73177784103', '50625303450', '99999000001']  # Example participants for testing (Singapore, London, Seattle)
test_circles = ['IP-SIN-01', 'IP-LON-04', 'IP-SEA-01']  # Test circles

# Define a general safe_string_match function at module level for use everywhere
def generate_circle_options_from_preferences(remaining_df, region, debug_mode=False):
    """
    Generate potential new circle options based on participant preferences.
    Used when no existing circles are found (greenfield scenarios).

    Args:
        remaining_df: DataFrame with participants needing placement
        region: Region name for the participants
        debug_mode: Whether to print debug information

    Returns:
        List of (subregion, time_slot) tuples representing potential circle options
    """

    # 🚀 CRITICAL DEBUG: Greenfield circle generation entry point
    print(f"\n🚀🚀🚀 GENERATE_CIRCLE_OPTIONS_FROM_PREFERENCES CALLED! 🚀🚀🚀")
    print(f"  Participants: {len(remaining_df)}")
    print(f"  Region: {region}")
    print(f"  Available columns: {remaining_df.columns.tolist()}")

    if debug_mode:
        print(f"\n🔧 Generating circle options from {len(remaining_df)} participant preferences in {region}")

    # Extract unique location preferences
    location_preferences = set()
    for col in ['first_choice_location', 'second_choice_location', 'third_choice_location']:
        if col in remaining_df.columns:
            locations = remaining_df[col].dropna().unique()
            location_preferences.update(locations)

    # Extract unique time preferences  
    time_preferences = set()
    for col in ['first_choice_time', 'second_choice_time', 'third_choice_time']:
        if col in remaining_df.columns:
            times = remaining_df[col].dropna().unique()
            time_preferences.update(times)

    # Remove empty values
    location_preferences = {loc for loc in location_preferences if loc and str(loc).strip()}
    time_preferences = {time for time in time_preferences if time and str(time).strip()}

    if debug_mode:
        print(f"🔧 Found {len(location_preferences)} unique location preferences: {list(location_preferences)}")
        print(f"🔧 Found {len(time_preferences)} unique time preferences: {list(time_preferences)}")

    # Generate circle options as cross-product of locations and times
    circle_options = []
    for location in location_preferences:
        for time_slot in time_preferences:
            circle_options.append((location, time_slot))

    if debug_mode:
        print(f"🔧 Created {len(circle_options)} potential circle options:")
        for i, (loc, time) in enumerate(circle_options[:5]):  # Show first 5
            print(f"  Option {i+1}: {loc} @ {time}")
        if len(circle_options) > 5:
            print(f"  ... and {len(circle_options) - 5} more options")

    return circle_options


def calculate_circle_diversity_score(participant_ids, results_df):
    """
    Calculate the total diversity score for a circle based on participant demographic data.
    Diversity score = sum of unique buckets across all 5 diversity categories.

    Args:
        participant_ids: List of participant Encoded IDs in the circle
        results_df: DataFrame containing participant demographic data

    Returns:
        int: Total diversity score (sum of vintage, employment, industry, racial identity, children scores)
    """
    if not participant_ids or results_df is None or len(participant_ids) == 0:
        return 0

    # Initialize sets to track unique categories
    unique_vintages = set()
    unique_employment = set()
    unique_industry = set()
    unique_racial_identity = set()
    unique_children = set()

    # Extract diversity data for each participant
    for participant_id in participant_ids:
        # Find participant data
        participant_data = results_df[results_df['Encoded ID'] == participant_id]
        if participant_data.empty:
            continue  # Ignore participants not found in results

        participant_row = participant_data.iloc[0]

        # Class Vintage diversity
        if 'Class_Vintage' in participant_data.columns:
            vintage = participant_row['Class_Vintage']
            if pd.notna(vintage):
                unique_vintages.add(vintage)

        # Employment diversity  
        if 'Employment_Category' in participant_data.columns:
            employment = participant_row['Employment_Category']
            if pd.notna(employment):
                unique_employment.add(employment)

        # Industry diversity
        if 'Industry_Category' in participant_data.columns:
            industry = participant_row['Industry_Category']
            if pd.notna(industry):
                unique_industry.add(industry)

        # Racial Identity diversity
        if 'Racial_Identity_Category' in participant_data.columns:
            racial_identity = participant_row['Racial_Identity_Category']
            if pd.notna(racial_identity):
                unique_racial_identity.add(racial_identity)

        # Children diversity
        if 'Children_Category' in participant_data.columns:
            children = participant_row['Children_Category']
            if pd.notna(children):
                unique_children.add(children)

    # Calculate individual category scores (number of unique buckets)
    vintage_score = len(unique_vintages)
    employment_score = len(unique_employment)
    industry_score = len(unique_industry)
    racial_identity_score = len(unique_racial_identity)
    children_score = len(unique_children)

    # Total diversity score is sum of all category scores
    total_diversity_score = (vintage_score + employment_score + industry_score + 
                           racial_identity_score + children_score)

    return total_diversity_score


def safe_string_match(value1, value2):
    """
    Safely compare two values that might be strings, numbers, or NaN.

    Args:
        value1: First value to compare
        value2: Second value to compare

    Returns:
        bool: True if values match or one is a prefix of the other, False otherwise
    """
    # Handle None values first
    if value1 is None or value2 is None:
        return False

    # Handle NaN values safely for both arrays and scalars
    try:
        # For array-like values, check if all are NaN
        isna1 = hasattr(pd.isna(value1), '__iter__') and pd.isna(value1).all() if hasattr(pd.isna(value1), '__iter__') else pd.isna(value1)
        isna2 = hasattr(pd.isna(value2), '__iter__') and pd.isna(value2).all() if hasattr(pd.isna(value2), '__iter__') else pd.isna(value2)

        if isna1 or isna2:
            return False
    except Exception as e:
        # If there's an error in checking NaN, log it and assume they don't match
        print(f"Warning: Error checking NaN in safe_string_match: {str(e)}")
        return False

    # Convert to string if needed
    str1 = str(value1) if not isinstance(value1, str) else value1
    str2 = str(value2) if not isinstance(value2, str) else value2

    # Exact match
    if str1 == str2:
        return True

    # Prefix match
    try:
        return str1.startswith(str2) or str2.startswith(str1)
    except (AttributeError, TypeError):
        # Extra safety in case conversion fails
        return False

def get_unique_preferences(df, columns):
    """
    Extract unique preference values from specified columns

    Args:
        df: DataFrame with participant data
        columns: List of column names to extract preferences from

    Returns:
        List of unique preference values
    """
    values = []
    for col in columns:
        if col in df.columns:
            values.extend(df[col].dropna().unique())
    return list(set(values))

def optimize_region_v2(region, region_df, min_circle_size, enable_host_requirement, existing_circle_handling, debug_mode=False):
    # 🔍 CRITICAL DEBUG: Function entry point logging
    print(f"\n🚀 OPTIMIZE_REGION_V2 CALLED!")
    print(f"  Region: {region}")
    print(f"  Participants: {len(region_df) if region_df is not None else 'None'}")
    print(f"  Min circle size: {min_circle_size}")
    print(f"  Existing circle handling: {existing_circle_handling}")
    print(f"  Debug mode: {debug_mode}")
    print(f"🔒 SAME-PERSON CONSTRAINT: Will be implemented in this function")

    # Import or define is_time_compatible here to ensure it's available in this scope
    # This fixes the "cannot access local variable" error in optimize mode
    from modules.data_processor import is_time_compatible

    # Import our new fixes module for CURRENT-CONTINUING members and optimize mode
    from modules.optimizer_fixes import (
        preprocess_continuing_members,
        preprocess_moving_into_region,
        optimize_circle_capacity,
        find_current_circle_id,
        force_compatibility,
        ensure_current_continuing_matched,
        post_process_continuing_members
    )

    # Import our new circle reconstruction module
    from modules.circle_reconstruction import reconstruct_circles_from_results

    # Import diagnostic tools for troubleshooting
    from modules.diagnostic_tools import (
        track_current_continuing_status,
        track_matching_outcomes,
        add_debug_constraints_log
    )

    # CRITICAL DIAGNOSTIC: Always enable debug mode for Seattle region
    if region == "Seattle":
        debug_mode = True

        # No longer forcing optimize mode - using the mode selected in the UI
        print(f"\n🔄 Seattle region is using '{existing_circle_handling}' mode as selected in UI")
        print(f"  'optimize' mode allows NEW participants to be matched with continuing circles like IP-SEA-01")
        print(f"  'preserve' mode prevents NEW participants from joining existing circles")
        print(f"  'dissolve' ignores current circles and creates all new ones")

        print(f"\n🔍 SEATTLE REGION DEEP DIAGNOSTICS:")
        print(f"  - Total participants in region_df: {len(region_df)}")

        # Count participants by status
        if 'Status' in region_df.columns:
            status_counts = region_df['Status'].value_counts().to_dict()
            print(f"  - Status counts: {status_counts}")
        else:
            print(f"  ⚠️ No 'Status' column found in region_df")

        # Analyze circle IDs
        if 'Current_Circle_ID' in region_df.columns:
            circle_ids = region_df['Current_Circle_ID'].dropna().unique()
            print(f"  - Existing circles: {list(circle_ids)}")

            # Check specifically for IP-SEA-01
            if 'IP-SEA-01' in circle_ids:
                circle_members = region_df[region_df['Current_Circle_ID'] == 'IP-SEA-01']
                print(f"  - IP-SEA-01 has {len(circle_members)} members")

                # Show member details
                for i, row in circle_members.iterrows():
                    print(f"    Member {row.get('Encoded ID')}:")
                    print(f"      Status: {row.get('Status')}")
                    print(f"      Time preferences: {row.get('first_choice_time')}, {row.get('second_choice_time')}, {row.get('third_choice_time')}")
            else:
                print(f"  ⚠️ IP-SEA-01 not found in circle_ids")
        else:
            print(f"  ⚠️ No 'Current_Circle_ID' column found in region_df")

        # Specifically check for NEW participants
        if 'Status' in region_df.columns:
            new_participants = region_df[region_df['Status'] == 'NEW']
            print(f"  - Found {len(new_participants)} NEW participants in Seattle")

            if len(new_participants) > 0:
                for i, row in new_participants.iterrows():
                    print(f"    NEW Participant {row.get('Encoded ID')}:")
                    print(f"      Location preferences: {row.get('first_choice_location')}, {row.get('second_choice_location')}, {row.get('third_choice_location')}")
                    print(f"      Time preferences: {row.get('first_choice_time')}, {row.get('second_choice_time')}, {row.get('third_choice_time')}")
            else:
                print(f"  🚨 CRITICAL ISSUE: No NEW participants found in Seattle region!")
    """
    Optimize matching within a single region using the refactored circle ID-based model

    Args:
        region: Region name
        region_df: DataFrame with participants from this region
        min_circle_size: Minimum number of participants per circle
        enable_host_requirement: Whether to enforce host requirements
        existing_circle_handling: How to handle existing circles ('preserve', 'dissolve', 'optimize')
        debug_mode: Whether to print debug information

    Returns:
        Tuple of (results list, circles list, unmatched list, debug_circles, circle_eligibility_logs)
    """
    # Initialize Seattle debug logs if we're processing Seattle region
    if region == "Seattle":
        # Set a flag to indicate we're in Seattle region
        is_seattle_region = True
        # Force debug mode for Seattle region
        debug_mode = True
        # Make sure logs are initialized in session state
        import streamlit as st
        if 'seattle_debug_logs' not in st.session_state:
            st.session_state.seattle_debug_logs = []
        # Record start of Seattle region analysis
        st.session_state.seattle_debug_logs.append(f"Starting Seattle region matching analysis")
        # Add participant counts
        st.session_state.seattle_debug_logs.append(f"Total participants in Seattle region: {len(region_df)}")
    # Define test participants for debugging purposes only (no special handling)
    test_participants = ['72549701782', '73177784103', '50625303450']
    test_circles = ['IP-HOU-02', 'IP-SIN-01', 'IP-LON-04']
    # Enable debug mode specifically for test case
    print("\n🔍 SPECIAL TEST CASE: Debugging participant 73177784103 match with circle IP-SIN-01 🔍")

    # CRITICAL FIX: Ensure test circles are always included in their respective regions
    # This addresses the issue where IP-SIN-01 wasn't available for matching with participant 73177784103
    if region == "Singapore":
        test_circle_exists = False
        for _, row in region_df.iterrows():
            if row.get("Current_Circle_ID") == "IP-SIN-01":
                test_circle_exists = True
                break

        if not test_circle_exists:
            print("\n🔧 CRITICAL FIX: Manually registering IP-SIN-01 in Singapore region")
            print("  This ensures the test circle is available for matching")
            # We'll handle this circle specially in the region filtering logic
    # Force debug mode to True for our critical test cases
    if region in ["London", "Singapore", "New York"]:
        debug_mode = True
        print(f"\n🔍🔍🔍 ENTERING CRITICAL REGION: {region} 🔍🔍🔍")

    # Check for any potential test data in the input
    print("\n🔍 INPUT DATA VALIDATION")

    # Filter for test participants (IDs starting with 99999)
    test_participant_ids = [p_id for p_id in region_df['Encoded ID'].values if str(p_id).startswith('99999')]
    if test_participant_ids:
        print(f"  ⚠️ Found {len(test_participant_ids)} potential test participants (IDs starting with 99999)")
        print(f"  These will be excluded from optimization results")

        # Remove test participants from the dataframe
        region_df = region_df[~region_df['Encoded ID'].astype(str).str.startswith('99999')]
        print(f"  ✅ Removed test participants from input data")
    else:
        print(f"  ✅ No test participants found in input data")

    # Check for test circles (containing TEST in the ID)
    test_circle_pattern = 'TEST'
    test_circles_found = False

    for col in region_df.columns:
        if 'circle' in col.lower() or 'id' in col.lower():
            circle_values = region_df[col].dropna().astype(str)
            test_circles = [c for c in circle_values if test_circle_pattern in c]
            if test_circles:
                test_circles_found = True
                print(f"  ⚠️ Found test circles in column {col}: {test_circles}")

    if not test_circles_found:
        print(f"  ✅ No test circles found in input data")

    # Print a notice about the new optimizer implementation
    print(f"\n🔄 Using new circle ID-based optimizer for region {region}")

    # Context information for enhanced unmatched reason determination
    optimization_context = {
        'existing_circles': [],
        'similar_participants': {},
        'full_circles': [],
        'circles_needing_hosts': [],
        'host_counts': {},
        'compatibility_matrix': {},  # Track compatibility between participants and circle options
        'participant_compatible_options': {},  # Count of compatible options per participant
        'location_time_pairs': []  # All possible location-time combinations
    }

    # Initialize containers for results
    results = []
    circles = []
    unmatched = []

    # Track debug information
    circle_capacity_debug = {}  # For tracking capacity of circles

    # Initialize local eligibility logs for this region
    # Instead of using a global, we'll create a local variable and return it
    circle_eligibility_logs = {}  # For tracking circle eligibility decisions

    # Debug message
    debug_eligibility_logs(f"Initialized local circle eligibility logs for {region} region")

    # Central registry to track processed circle IDs and prevent duplicates
    processed_circle_ids = set()

    # Track timing for performance analysis
    start_time = time.time()

    # Initialize containers for existing circle handling
    existing_circles = {}  # Maps circle_id to circle data for viable circles (>= min_circle_size)
    small_circles = {}     # Maps circle_id to circle data for small circles (2-4 members)
    current_circle_members = {}  # Maps circle_id to list of members

    # Step 1: Identify existing circles based on existing_circle_handling mode
    # Print clear debug message about which mode we're using (especially for Seattle)
    print(f"\n🔥 PROCESSING REGION '{region}' WITH existing_circle_handling='{existing_circle_handling}'") 

    if region == "Seattle":
        print(f"  This means NEW participants CAN be matched with existing circles: {existing_circle_handling == 'optimize'}")
        print(f"  For Seattle region, we need 'optimize' mode to allow NEW participants to match with IP-SEA-01")

    # Check for circle ID column (case-insensitive to handle column mapping issues) for all modes
    # We need this column for both 'preserve' and 'optimize' modes
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID']

    # Print potential column names for debugging
    if debug_mode:
        print(f"Looking for circle ID column. Options: {potential_columns}")
        print(f"Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if col.lower() in [c.lower() for c in potential_columns]:
                current_col = col
                break

    if current_col is None and debug_mode:
        print(f"CRITICAL ERROR: Could not find current circles ID column. Available columns: {region_df.columns.tolist()}")
        return [], [], [], {}, {}  # Return empty results if we can't find the critical column

    # Now process based on the selected mode
    if existing_circle_handling == 'preserve' or existing_circle_handling == 'optimize':

        if current_col is not None:
            if debug_mode:
                print(f"Using column '{current_col}' for current circle IDs")
                continuing_count = len(region_df[region_df['Status'] == 'CURRENT-CONTINUING'])
                circles_count = region_df[region_df['Status'] == 'CURRENT-CONTINUING'][current_col].notna().sum()
                print(f"Found {continuing_count} CURRENT-CONTINUING participants, {circles_count} with circle IDs")

            # Group participants by their current circle
            current_continuing_with_problems = []

            for _, row in region_df.iterrows():
                # First, check if it's a CURRENT-CONTINUING participant
                if row.get('Status') == 'CURRENT-CONTINUING':
                    # They must have a current circle ID - this is required for CURRENT-CONTINUING
                    if pd.notna(row.get(current_col)):
                        circle_id = str(row[current_col]).strip()
                        if circle_id:
                            if circle_id not in current_circle_members:
                                current_circle_members[circle_id] = []
                            current_circle_members[circle_id].append(row)

                            # CRITICAL FIX: Log successful assignments of CURRENT-CONTINUING members
                            if debug_mode:
                                print(f"✅ CURRENT-CONTINUING participant {row['Encoded ID']} assigned to {circle_id}")
                        else:
                            # They're CURRENT-CONTINUING but have an empty circle ID
                            # This shouldn't happen per the spec, but log for debugging
                            if debug_mode:
                                print(f"⚠️ WARNING: CURRENT-CONTINUING participant {row['Encoded ID']} has empty circle ID")
                            current_continuing_with_problems.append(row)
                    else:
                        # They're CURRENT-CONTINUING but circle ID is null
                        # This shouldn't happen per the spec, but log for debugging
                        if debug_mode:
                            print(f"⚠️ WARNING: CURRENT-CONTINUING participant {row['Encoded ID']} has null circle ID")
                        current_continuing_with_problems.append(row)

            # CRITICAL FIX: Try harder to find circle IDs for problematic CURRENT-CONTINUING participants
            if current_continuing_with_problems:
                print(f"🚨 Found {len(current_continuing_with_problems)} CURRENT-CONTINUING participants with missing circle IDs")
                print(f"🔍 Attempting alternative methods to find their circle IDs...")

                # Try checking other columns for circle IDs
                for i, problem_row in enumerate(current_continuing_with_problems):
                    p_id = problem_row['Encoded ID']
                    found_circle = None

                    # Check all columns for anything that looks like a circle ID
                    for col in problem_row.index:
                        if pd.notna(problem_row[col]) and isinstance(problem_row[col], str):
                            col_value = str(problem_row[col]).strip()
                            # Check if it matches circle ID pattern (e.g., IP-NYC-18)
                            if '-' in col_value and any(c.isalpha() for c in col_value) and any(c.isdigit() for c in col_value):
                                potential_circle = col_value
                                print(f"  👉 Found potential circle ID '{potential_circle}' for {p_id} in column '{col}'")
                                found_circle = potential_circle
                                break

                    if found_circle:
                        print(f"  ✅ Recovered circle ID '{found_circle}' for CURRENT-CONTINUING participant {p_id}")
                        if found_circle not in current_circle_members:
                            current_circle_members[found_circle] = []
                        current_circle_members[found_circle].append(problem_row)
                    else:
                        print(f"  ❌ Failed to recover circle ID for CURRENT-CONTINUING participant {p_id}")

        # Evaluate each existing circle in the region
        # Note: By this point, direct continuation has already been done in the main function
        # so we only need to handle edge cases here
        print(f"\n🔍 DEBUG: Processing {len(current_circle_members)} existing circles in {region} region")
        print(f"Circle IDs: {list(current_circle_members.keys())}")

        for circle_id, members in current_circle_members.items():
            # Per PRD: An existing circle is maintained if it has at least 2 CURRENT-CONTINUING members
            # and meets host requirements (for in-person circles)
            print(f"\n👉 Processing circle {circle_id} with {len(members)} members")
            if len(members) >= 2:
                # Check if it's an in-person circle (IP prefix) or virtual circle (V prefix)
                is_in_person = circle_id.startswith('IP-') and not circle_id.startswith('IP-NEW-')
                is_virtual = circle_id.startswith('V-') and not circle_id.startswith('V-NEW-')

                # For in-person circles, check host requirements
                host_requirement_met = True
                if is_in_person and enable_host_requirement:
                    has_host = any(m.get('host', '').lower() in ['always', 'always host', 'sometimes', 'sometimes host'] for m in members)
                    host_requirement_met = has_host

                if host_requirement_met:
                    # IMPROVED: Get subregion and time by checking all members of the circle, not just the first one
                    # Initialize with fallback values
                    subregion = "Unknown"
                    meeting_day = ""
                    meeting_time = ""

                    # Possible column names for subregion
                    subregion_columns = ['Current_Subregion', 'Current Subregion', 'Current/ Continuing Subregion', 'Current_Region']

                    # Possible column names for meeting day
                    day_columns = ['Current_Meeting_Day', 'Current Meeting Day', 'Current/ Continuing Meeting Day']

                    # Possible column names for meeting time
                    time_columns = ['Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time']

                    # Iterate through all members to find valid subregion and meeting time info
                    for member in members:
                        # Try to find subregion
                        if not subregion or subregion == "Unknown":
                            for col in subregion_columns:
                                if col in member and pd.notna(member.get(col)) and member.get(col):
                                    subregion = str(member.get(col))
                                    if debug_mode:
                                        print(f"  Found subregion '{subregion}' for circle {circle_id} from column {col}")
                                    break

                        # Try to find meeting day
                        if not meeting_day:
                            for col in day_columns:
                                if col in member and pd.notna(member.get(col)) and member.get(col):
                                    meeting_day = str(member.get(col))
                                    if debug_mode:
                                        print(f"  Found meeting day '{meeting_day}' for circle {circle_id} from column {col}")
                                    break

                        # Try to find meeting time
                        if not meeting_time:
                            for col in time_columns:
                                if col in member and pd.notna(member.get(col)) and member.get(col):
                                    meeting_time = str(member.get(col))
                                    if debug_mode:
                                        print(f"  Found meeting time '{meeting_time}' for circle {circle_id} from column {col}")
                                    break

                        # If we have all the information we need, we can break out of the loop
                        if subregion != "Unknown" and meeting_day and meeting_time:
                            break

                    # Standardize time format to plural form (e.g., "Evening" -> "Evenings", "Day" -> "Days")
                    # This ensures consistency between existing circles and new participant preferences
                    if meeting_time.lower() == 'evening':
                        meeting_time = 'Evenings'
                    elif meeting_time.lower() == 'day':
                        meeting_time = 'Days'

                    # Format the day/time combination
                    if meeting_day and meeting_time:
                        formatted_meeting_time = f"{meeting_day} ({meeting_time})"
                    elif meeting_day:
                        formatted_meeting_time = meeting_day
                    elif meeting_time:
                        formatted_meeting_time = meeting_time
                    else:
                        formatted_meeting_time = "Unknown"  # Use "Unknown" as fallback for the optimizer

                    if debug_mode:
                        print(f"For existing circle_id {circle_id}, using day '{meeting_day}' and time '{meeting_time}'")
                        print(f"Formatted meeting time: '{formatted_meeting_time}'")

                    # Calculate max_additions for this circle based on co-leader preferences
                    max_additions = None  # Start with None to indicate no limit specified yet
                    has_none_preference = False
                    has_co_leader = False

                    for member in members:
                        # Only consider preferences from current co-leaders
                        # Try to find the co-leader column with the correct name
                        co_leader_value = ''
                        if 'Current Co-Leader?' in member:
                            co_leader_value = str(member.get('Current Co-Leader?', ''))
                        elif 'Current_Co_Leader' in member:
                            co_leader_value = str(member.get('Current_Co_Leader', ''))

                        is_current_co_leader = co_leader_value.strip().lower() == 'yes'

                        if debug_mode and circle_id in ['IP-SIN-01', 'IP-LON-04']:
                            print(f"  Co-Leader check in existing circle preservation: value='{co_leader_value}', is_co_leader={is_current_co_leader}")

                        # Skip non-co-leaders
                        if not is_current_co_leader:
                            continue

                        # Mark that at least one co-leader was found
                        has_co_leader = True

                        # Get the max new members value if present
                        max_value = member.get('co_leader_max_new_members', None)

                        if debug_mode and pd.notna(max_value):
                            print(f"  Co-Leader {member['Encoded ID']} specified max new members: {max_value}")

                        # With our new standardized approach, max_value is always a string
                        # Check for "None" value (will always be string "None" now)
                        if max_value == "None":
                            has_none_preference = True
                            if debug_mode:
                                print(f"  Co-Leader {member['Encoded ID']} specified 'None' - no new members allowed")
                            break

                        # Process numeric values - all valid numbers are string representations now
                        elif pd.notna(max_value):
                            try:
                                int_value = int(max_value)  # Convert string to int
                                # If first valid value or smaller than previous minimum
                                if max_additions is None or int_value < max_additions:
                                    max_additions = int_value
                                    if debug_mode:
                                        print(f"  Co-Leader {member['Encoded ID']} specified {int_value} max new members")
                            except (ValueError, TypeError):
                                # Not a valid number, log error and treat as None
                                if debug_mode:
                                    print(f"  Invalid max new members value: {max_value}, treating as 'None'")
                                has_none_preference = True
                                break

                    # Set max_additions based on rules
                    if has_none_preference:
                        # Determine the current size of the circle
                        circle_size = len(members)

                        # DEBUG for checking "None" preference handling
                        global DEBUG_ELIGIBILITY_COUNTER
                        DEBUG_ELIGIBILITY_COUNTER += 1

                        print(f"\n🔍 DEBUG CIRCLE ELIGIBILITY #{DEBUG_ELIGIBILITY_COUNTER} 🔍")
                        print(f"Circle {circle_id} with {circle_size} members has 'None' preference")
                        print(f"  Small circle? {circle_size < 5}")
                        print(f"  Region: {region}")

                        # UNIVERSAL FIX: Allow small circles to grow regardless of co-leader preferences
                        if circle_size < 5:  # 5 is the minimum viable size
                            # Override the "None" preference to allow smaller circles to reach viable size
                            # Universal fix for small circles: add enough to reach viable size
                            final_max_additions = max(0, 5 - circle_size)

                            # If circle has 4 members, ensure they can add at least 1 to reach viable size
                            if circle_size == 4:
                                final_max_additions = max(1, final_max_additions)

                            # Ensure we don't exceed configurable maximum total members for any circle
                            import streamlit as st
                            max_circle_size = st.session_state.get('max_circle_size', 8) if 'st' in globals() else 8
                            final_max_additions = min(final_max_additions, max_circle_size - circle_size)

                            if final_max_additions > 0:
                                print(f"  🔷 UNIVERSAL FIX APPLIED: Small circle {circle_id} with {circle_size} members")
                                print(f"  Setting max_additions={final_max_additions} to help reach viable size")

                            # Record why we're making this change in our logs
                            circle_eligibility_logs[circle_id] = {
                                'circle_id': circle_id,
                                'region': region,
                                'subregion': subregion,
                                'meeting_time': formatted_meeting_time,
                                'max_additions': final_max_additions,
                                'current_members': circle_size,
                                'is_eligible': final_max_additions > 0,
                                'original_preference': 'None',
                                'override_reason': 'Small circle needs to reach viable size',
                                'is_test_circle': False,
                                'is_small_circle': circle_size < 5,
                                'has_none_preference': True,
                                'preference_overridden': True
                            }

                            if debug_mode and final_max_additions > 0:
                                print(f"  ✅ SMALL CIRCLE ELIGIBILITY: {circle_id} can accept {final_max_additions} new members")
                        else:
                            # Regular case: Co-leader requested no new members and circle is already viable
                            final_max_additions = 0

                            # Record why this circle is excluded
                            circle_eligibility_logs[circle_id] = {
                                'circle_id': circle_id,
                                'region': region,
                                'subregion': subregion,
                                'meeting_time': formatted_meeting_time,
                                'max_additions': final_max_additions,
                                'current_members': circle_size,
                                'is_eligible': False,
                                'original_preference': 'None',
                                'reason': 'Co-leader requested no new members and circle is already viable',
                                'is_test_circle': False,
                                'is_small_circle': circle_size < 5,
                                'has_none_preference': True,
                                'preference_overridden': False
                            }

                            if debug_mode:
                                print(f"  Circle {circle_id} has 'None' preference from co-leader - not accepting new members")


                    elif max_additions is not None:
                        # Use the minimum valid value provided by co-leaders
                        # BUT cap it to respect the configured maximum circle size
                        import streamlit as st
                        max_circle_size = st.session_state.get('max_circle_size', 8) if 'st' in globals() else 8
                        current_members = len(members)
                        max_allowed_additions = max(0, max_circle_size - current_members)

                        # Cap co-leader preference to respect configured maximum
                        original_preference = max_additions
                        final_max_additions = min(max_additions, max_allowed_additions)

                        # Log when co-leader preference is overridden by maximum circle size
                        preference_overridden = final_max_additions < original_preference
                        if preference_overridden and debug_mode:
                            print(f"  ⚠️ Co-leader preference capped: {circle_id} requested {original_preference} but limited to {final_max_additions} (max size: {max_circle_size})")

                        # Record this circle in the eligibility logs
                        circle_eligibility_logs[circle_id] = {
                            'circle_id': circle_id,
                            'region': region,
                            'subregion': subregion,
                            'meeting_time': formatted_meeting_time,
                            'max_additions': final_max_additions,
                            'current_members': len(members),
                            'is_eligible': final_max_additions > 0,
                            'original_preference': 'Specified by co-leader',
                            'preference_value': original_preference,
                            'is_test_circle': False,
                            'is_small_circle': len(members) < 5,
                            'has_none_preference': False,
                            'preference_overridden': preference_overridden
                        }

                        if debug_mode:
                            print(f"  Circle {circle_id} can accept up to {final_max_additions} new members (co-leader preference)")
                    else:
                        # Default to configured maximum if no co-leader specified a value or no co-leaders exist
                        import streamlit as st
                        max_circle_size = st.session_state.get('max_circle_size', 8) if 'st' in globals() else 8
                        final_max_additions = max(0, max_circle_size - len(members))

                        # Record this circle in the eligibility logs
                        circle_eligibility_logs[circle_id] = {
                            'circle_id': circle_id,
                            'region': region,
                            'subregion': subregion,
                            'meeting_time': formatted_meeting_time,
                            'max_additions': final_max_additions,
                            'current_members': len(members),
                            'is_eligible': final_max_additions > 0,
                            'original_preference': 'Default',
                            'preference_value': max_circle_size,  # Use configurable maximum instead of hardcoded 8
                            'is_test_circle': False,
                            'is_small_circle': len(members) < 5,
                            'has_none_preference': False,
                            'preference_overridden': False
                        }

                        if debug_mode:
                            message = "No co-leader preference specified" if has_co_leader else "No co-leaders found"
                            print(f"  {message} for circle {circle_id} - using default max total of 8")
                            print(f"  Currently has {len(members)} members, can accept {final_max_additions} more")

                    # Use the centralized region extraction utility
                    normalized_current_region = normalize_region_name(region)

                    # Extract and normalize the circle's region using our utility function
                    circle_region = extract_region_code_from_circle_id(circle_id)

                    # If region extraction failed, get the region from a representative circle member
                    if circle_region is None and len(members) > 0:
                        # Try to get region from the first member
                        circle_region = get_region_from_circle_or_participant(members[0])

                    # Last resort: default to current region if we still couldn't determine it
                    if circle_region is None:
                        circle_region = region

                    # Normalize the circle_region for consistent comparison
                    circle_region = normalize_region_name(circle_region)

                    # DETAILED DEBUG: Enhanced region analysis
                    if TRACE_REGION_MAPPING or debug_mode:
                        print(f"\n🔍 REGION MAPPING: Circle {circle_id}")
                        print(f"  Current processing region: {region} (normalized: {normalized_current_region})")
                        print(f"  Circle region: {circle_region}")


                    # Determine if this circle should be skipped in this region
                    circle_should_be_skipped = False

                    # Use normalized region comparison for all circles
                    if circle_region != normalized_current_region:
                        circle_should_be_skipped = True
                        if debug_mode:
                            print(f"  📍 Region mismatch: Circle {circle_id} belongs to {circle_region}, not {normalized_current_region}")

                    # Skip this circle if it doesn't belong to the current region
                    if circle_should_be_skipped:
                        if debug_mode:
                            print(f"  ⏩ Skipping circle {circle_id} in region {region} - belongs to region {circle_region}")
                        continue

                    # Create the circle data
                    circle_data = {
                        'circle_id': circle_id,
                        'region': region,
                        'subregion': subregion,
                        'meeting_time': formatted_meeting_time,
                        'members': [member['Encoded ID'] for member in members],
                        'member_count': len(members),
                        'max_additions': final_max_additions,
                        'is_existing': True
                    }

                    # Count hosts in existing circle
                    circle_data['always_hosts'] = sum(1 for m in members 
                                                  if m.get('host', '').lower() in ['always', 'always host'])
                    circle_data['sometimes_hosts'] = sum(1 for m in members 
                                                      if m.get('host', '').lower() in ['sometimes', 'sometimes host'])

                    existing_circles[circle_id] = circle_data

                    # Add to circles list if it has 5+ members (it's already viable)
                    if len(members) >= 5:
                        circle_copy = circle_data.copy()
                        circle_copy['is_continuing'] = True
                        circle_copy['new_members'] = 0
                        # Only add if it's not already in the registry
                        if circle_id not in processed_circle_ids:
                            circles.append(circle_copy)
                            processed_circle_ids.add(circle_id)
                            if debug_mode:
                                print(f"  Added existing viable circle {circle_id} to results (initial processing)")
                    # Otherwise add to small circles list (2-4 members)
                    elif len(members) >= 2 and len(members) <= 4:
                        small_circles[circle_id] = circle_data

    # CRITICAL FIX: If we didn't find any real existing circles, create synthetic test circles
    # to verify that our eligibility logging works correctly
    if not existing_circles:
        print(f"\n🚨 CRITICAL ISSUE: No existing circles found for region {region}")
        print(f"🔍 Analyzing region_df for potential circles:")

        # DEEP DIAGNOSTICS: Check the data structure in full detail
        print(f"\n🔬 DEEP DATA DIAGNOSTICS FOR REGION {region}")
        print(f"🔬 DataFrame shape: {region_df.shape}")

        # Check status distribution
        print(f"🔬 Status column values:")
        if 'Status' in region_df.columns:
            status_counts = region_df['Status'].value_counts().to_dict()
            print(f"   Status counts: {status_counts}")
        else:
            print(f"   'Status' column not found in DataFrame")

        # Check for continuing participants
        if 'Status' in region_df.columns:
            continuing = region_df[region_df['Status'] == 'CURRENT-CONTINUING']
            print(f"🔬 CURRENT-CONTINUING participants: {len(continuing)}")

            # If we have continuing participants, check their circle IDs
            if len(continuing) > 0:
                # Check all potential circle ID columns
                circle_columns = ['Current_Circle_ID', 'current_circles_id', 'Current Circle ID']
                for col in circle_columns:
                    if col in continuing.columns:
                        valid_ids = continuing[~continuing[col].isna()]
                        if len(valid_ids) > 0:
                            print(f"🔬 Found {len(valid_ids)} participants with non-null '{col}' values")
                            unique_circles = valid_ids[col].unique()
                            print(f"🔬 Unique circle IDs: {len(unique_circles)}")
                            print(f"🔬 Sample circle IDs: {list(unique_circles)[:5]}{'...' if len(unique_circles) > 5 else ''}")

                            # Find the first few participants for sample circle ID
                            if len(unique_circles) > 0:
                                sample_circle = unique_circles[0]
                                sample_members = continuing[continuing[col] == sample_circle]
                                print(f"\n🔬 DETAILED INSPECTION OF CIRCLE {sample_circle}:")
                                print(f"   Members: {len(sample_members)}")

                                # Print full details of a sample participant
                                if len(sample_members) > 0:
                                    sample_member = sample_members.iloc[0]
                                    print(f"\n🔬 SAMPLE MEMBER COMPLETE DATA FOR DIAGNOSTIC:")
                                    for c, val in sample_member.items():
                                        print(f"   {c}: {val}")

            # Check meeting time and subregion columns
            meeting_day_col = None
            meeting_time_col = None
            subregion_col = None

            # Find meeting day column
            for col in ['Current_Meeting_Day', 'Current Meeting Day', 'current_meeting_day']:
                if col in region_df.columns:
                    meeting_day_col = col
                    break

            # Find meeting time column
            for col in ['Current_Meeting_Time', 'Current Meeting Time', 'current_meeting_time']:
                if col in region_df.columns:
                    meeting_time_col = col
                    break

            # Find subregion column
            for col in ['Current_Subregion', 'Current Subregion', 'current_subregion']:
                if col in region_df.columns:
                    subregion_col = col
                    break

            # Print what we found
            print(f"\n🔬 CRITICAL COLUMN DETECTION:")
            print(f"   Meeting day column: {meeting_day_col}")
            print(f"   Meeting time column: {meeting_time_col}")
            print(f"   Subregion column: {subregion_col}")

        # Check if we have Circle ID column
        current_col = None
        potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID']

        for col in potential_columns:
            if col in region_df.columns:
                current_col = col
                break

        # If we found the column, print detailed stats
        if current_col:
            print(f"✅ Found circle ID column: {current_col}")

            # Check counts by Status
            status_counts = region_df['Status'].value_counts()
            print(f"Status counts: {status_counts.to_dict()}")

            # Check for CURRENT-CONTINUING participants
            continuing = region_df[region_df['Status'] == 'CURRENT-CONTINUING']
            print(f"Found {len(continuing)} CURRENT-CONTINUING participants")

            # Check how many have circle IDs
            with_circles = continuing[pd.notna(continuing[current_col])]
            print(f"Of those, {len(with_circles)} have a non-null circle ID")

            # Show the first few circle IDs
            if len(with_circles) > 0:
                circle_ids = with_circles[current_col].unique()
                print(f"Circle IDs found: {list(circle_ids)[:5]}{'...' if len(circle_ids) > 5 else ''}")

                # CRITICAL ROOT CAUSE FIX: Create real circles from existing data
                print(f"IMPLEMENTING ROOT CAUSE FIX: Creating real circles from participant data")

                # Process every circle ID we found
                real_circles = {}
                circles_created = 0

                for circle_id in circle_ids:
                    # Skip empty or NaN values
                    if pd.isna(circle_id) or str(circle_id).strip() == '':
                        continue

                    # Get all members of this circle
                    circle_members = with_circles[with_circles[current_col] == circle_id]
                    member_count = len(circle_members)

                    # Only process circles with at least 2 members (per business rules)
                    if member_count >= 2:
                        circles_created += 1

                        # Find subregion if available - checking all possible column names
                        subregion = "Unknown"
                        for subregion_col in ['Current_Subregion', 'Current Subregion', 'Current/ Continuing Subregion']:
                            if subregion_col in circle_members.columns:
                                subregions = circle_members[subregion_col].dropna().unique()
                                if len(subregions) > 0:
                                    subregion = str(subregions[0])
                                    break

                        # Find meeting day and time components separately
                        meeting_day = ""
                        meeting_time_value = ""

                        # Look for meeting day
                        for day_col in ['Current_Meeting_Day', 'Current Meeting Day', 'Current/ Continuing Meeting Day']:
                            if day_col in circle_members.columns:
                                days = circle_members[day_col].dropna().unique()
                                if len(days) > 0:
                                    meeting_day = str(days[0])
                                    break

                        # Look for meeting time
                        for time_col in ['Current_Meeting_Time', 'Current Meeting Time', 'Current/ Continuing Meeting Time']:
                            if time_col in circle_members.columns:
                                times = circle_members[time_col].dropna().unique()
                                if len(times) > 0:
                                    meeting_time_value = str(times[0])
                                    break

                        # Also check directly for combined day/time columns
                        combined_meeting_time = ""
                        for combined_col in ['Current_DayTime', 'Current/ Continuing DayTime', 'proposed_NEW_DayTime']:
                            if combined_col in circle_members.columns:
                                combined_times = circle_members[combined_col].dropna().unique()
                                if len(combined_times) > 0:
                                    combined_meeting_time = str(combined_times[0])
                                    break

                        # Now determine the final meeting_time to use
                        if combined_meeting_time:
                            # Use directly combined value if available
                            meeting_time = combined_meeting_time
                        elif meeting_day and meeting_time_value:
                            # Combine day and time components
                            meeting_time = f"{meeting_day} ({meeting_time_value})"
                        elif meeting_day:
                            # Just use day
                            meeting_time = meeting_day
                        elif meeting_time_value:
                            # Just use time
                            meeting_time = meeting_time_value
                        else:
                            # Default fallback
                            meeting_time = "Unknown"

                        # Process co-leader preference for max additions
                        max_additions = None
                        for pref_col in ['co_leader_max_new_members', 'Current_Co_Leader']:
                            if pref_col in circle_members.columns:
                                prefs = circle_members[pref_col].dropna().unique()
                                if len(prefs) > 0:
                                    try:
                                        # Try to interpret as a number if possible
                                        pref = prefs[0]
                                        if isinstance(pref, (int, float)) and not pd.isna(pref):
                                            max_additions = int(pref)
                                        elif isinstance(pref, str) and pref.strip() not in ['', 'None', 'nan', 'N/A']:
                                            try:
                                                max_additions = int(pref.strip())
                                            except ValueError:
                                                pass
                                    except:
                                        pass

                        # Default logic: 5-8 members total is normal target
                        # Small circles (2-4 members) need to get to at least 5
                        # Circles with 5+ members can accept up to 8 total
                        if max_additions is None:
                            if member_count < 5:
                                # Small circle - aim for at least 5
                                max_additions = 5 - member_count
                            else:
                                # Normal circle - max 8 total
                                max_additions = 8 - member_count
                                max_additions = max(0, max_additions)  # Ensure non-negative

                        # Create circle data structure
                        real_circles[str(circle_id)] = {
                            'circle_id': str(circle_id),
                            'region': region,
                            'subregion': subregion,
                            'meeting_time': meeting_time,
                            'member_count': member_count,
                            'members': circle_members['Encoded ID'].tolist(),
                            'is_existing': True,
                            'max_additions': max(0, max_additions),  # Ensure non-negative
                            'always_hosts': 1,  # Assume at least one host 
                            'sometimes_hosts': 0
                        }

                        print(f"  Created real circle {circle_id} with {member_count} members, max_additions={max_additions}")

                        # Limit to first 10 circles to avoid excessive log volume
                        if circles_created >= 10:
                            print(f"  ... and more (showing first 10 of {len(circle_ids)} circles)")
                            break

                # Use these real circles as our existing circles
                if real_circles:
                    print(f"🔧 Added {len(real_circles)} real circles from participant data!")
                    existing_circles = real_circles

            else:
                print(f"⚠️ All CURRENT-CONTINUING participants have null circle IDs")
        else:
            print(f"❌ Could not find circle ID column in region_df")
            print(f"Available columns: {region_df.columns.tolist()}")

    # Initialize new_circle_options early and generate immediately for greenfield scenarios
    new_circle_options = []

    # DEBUG: Check the conditions for circle generation
    print(f"\n🔍 DEBUG - Circle generation conditions:")
    print(f"  existing_circles count: {len(existing_circles) if existing_circles else 'None/Empty'}")
    print(f"  region_df size: {len(region_df)}")
    print(f"  min_circle_size: {min_circle_size}")
    print(f"  Condition 'not existing_circles': {not existing_circles}")
    print(f"  Condition 'len(region_df) >= min_circle_size': {len(region_df) >= min_circle_size}")

    # Generate new circle options immediately if no existing circles (greenfield scenario)
    if not existing_circles and len(region_df) >= min_circle_size:
        print(f"\n🔧 No existing circles found - generating new circle options from participant preferences")
        new_circle_options = generate_circle_options_from_preferences(region_df, region, debug_mode)
        print(f"🔧 Generated {len(new_circle_options)} potential new circle options")
    elif not existing_circles:
        print(f"\n🔧 No existing circles found, but insufficient participants ({len(region_df)}) to create new circles (min: {min_circle_size})")
    else:
        print(f"\n🔧 Existing circles found ({len(existing_circles)}), skipping new circle generation")

    # STEP 1.5: Preprocess MOVING INTO Region participants
    # Auto-fill first_choice_location with Current_Subregion if blank
    region_df = preprocess_moving_into_region(region_df, debug_mode=debug_mode)

    # For continuing participants not in circles, we need to handle them separately
    remaining_participants = []

    for _, row in region_df.iterrows():
        participant_id = row['Encoded ID']
        # Check if this participant is in any of our existing circles
        in_existing_circle = False
        for circle_data in existing_circles.values():
            if participant_id in circle_data['members']:
                in_existing_circle = True
                break

        # If not in any circle, add to remaining participants for matching
        if not in_existing_circle:
            remaining_participants.append(participant_id)

    # Create a DataFrame of just the remaining participants
    remaining_df = region_df[region_df['Encoded ID'].isin(remaining_participants)].copy()

    if debug_mode:
        print(f"After processing existing circles:")
        print(f"  {len(existing_circles)} viable circles with 5+ members")
        print(f"  {len(small_circles)} small circles with 2-4 members")
        print(f"  {len(new_circle_options)} new circle options generated")
        print(f"  {len(remaining_participants)} participants remain to be matched")

    # Handle case with no participants to match
    if len(remaining_participants) == 0:
        if debug_mode:
            print("No remaining participants to match. Returning existing circles.")

        # Create results for all participants in existing circles
        for circle_id, circle_data in existing_circles.items():
            for participant_id in circle_data['members']:
                participant = region_df[region_df['Encoded ID'] == participant_id].iloc[0].to_dict()
                participant['proposed_NEW_circles_id'] = circle_id
                participant['proposed_NEW_Subregion'] = circle_data['subregion']
                participant['proposed_NEW_DayTime'] = circle_data['meeting_time']
                participant['unmatched_reason'] = ""

                # Default scores for existing circle members
                participant['location_score'] = 3  # Assume max score for simplicity
                participant['time_score'] = 3      # Assume max score for simplicity
                participant['total_score'] = 6     # Sum of above

                results.append(participant)

        return results, circles, [], circle_capacity_debug, circle_eligibility_logs

    # Get all unique subregions and time slots for preference matching
    subregions = get_unique_preferences(remaining_df, ['first_choice_location', 'second_choice_location', 'third_choice_location'])
    time_slots = get_unique_preferences(remaining_df, ['first_choice_time', 'second_choice_time', 'third_choice_time'])

    # Filter out empty subregions/time slots
    subregions = [s for s in subregions if s]
    time_slots = [t for t in time_slots if t]

    # Store for use in unmatched reason determination
    optimization_context['subregions'] = subregions
    optimization_context['time_slots'] = time_slots

    # Get all viable circles with capacity for new members
    # Log why circles are or aren't viable
    print(f"\n🔍 CRITICAL DEBUG: Processing existing_circles for region {region}")
    print(f"🔍 Found {len(existing_circles)} circles in region {region} to evaluate for eligibility")

    # CRITICAL DEBUG: Print comprehensive information about existing circles
    if len(existing_circles) == 0:
        print(f"❌ CRITICAL ISSUE: No existing circles found for region {region}! This explains missing eligibility logs.")
        print(f"🔍 Check how existing_circles gets populated for this region")
    else:
        print(f"🔍 Circle IDs: {list(existing_circles.keys())[:5]}{'...' if len(existing_circles) > 5 else ''}")
        print(f"🔍 DETAILED EXAMINATION OF FIRST CIRCLE:")
        first_circle_id = list(existing_circles.keys())[0]
        first_circle = existing_circles[first_circle_id]
        print(f"Circle ID: {first_circle_id}")
        for key, value in first_circle.items():
            print(f"  {key}: {value}")

    # DEBUG: Show what the local circle_eligibility_logs contains before we start
    print(f"🔍 Before adding new logs, circle_eligibility_logs has {len(circle_eligibility_logs)} entries")

    # Counter for tracking how many circles we're processing
    circles_processed = 0

    for circle_id, circle_data in existing_circles.items():
        max_additions = circle_data.get('max_additions', 0)
        is_viable = max_additions > 0

        # Track detailed eligibility info
        circle_eligibility_logs[circle_id] = {
            'circle_id': circle_id,
            'region': circle_data.get('region', 'Unknown'),
            'subregion': circle_data.get('subregion', 'Unknown'),
            'meeting_time': circle_data.get('meeting_time', 'Unknown'),
            'max_additions': max_additions,
            'current_members': circle_data.get('member_count', 0),
            'is_eligible': is_viable,
            'reason': "Has capacity" if is_viable else "No capacity (max_additions=0)",
            'is_test_circle': False, # No test circles in the system
            'is_small_circle': circle_data.get('member_count', 0) < 5,
            'has_none_preference': max_additions == 0,  # Infer that 0 max_additions likely means "None" preference
            'preference_overridden': False  # By this point, overrides have already been applied above
        }

        # Add immediate verification that the log was created
        print(f"✅ CIRCLE LOG CREATED: {circle_id} → {circle_eligibility_logs[circle_id]['is_eligible']}")

        # Verify the global variable is being updated
        assert circle_id in circle_eligibility_logs, f"Failed to add {circle_id} to circle_eligibility_logs!"

        # Print detailed log for first few circles
        circles_processed += 1
        if circles_processed <= 3:
            print(f"🔍 Added eligibility log for circle {circle_id}:")
            print(f"   Region: {circle_data.get('region', 'Unknown')}")
            print(f"   Max Additions: {max_additions}, Is Viable: {is_viable}")
            print(f"   Current Members: {circle_data.get('member_count', 0)}")

    # After processing all circles, print a summary
    print(f"🔍 Finished processing {circles_processed} circles for eligibility")
    print(f"🔍 After processing, circle_eligibility_logs now has {len(circle_eligibility_logs)} entries")

    # CRITICAL FIX: Mark this section as being fixed to confirm all circles were processed
    print("\n🚨 CRITICAL FIX CONFIRMATION")
    print(f"✅ Successfully processed ALL {len(existing_circles)} circles in region {region}")

    # Debug verification of circle eligibility logs
    eligible_count = sum(1 for log in circle_eligibility_logs.values() if log.get('is_eligible', False))
    small_count = sum(1 for log in circle_eligibility_logs.values() if log.get('is_small_circle', False))
    test_count = sum(1 for log in circle_eligibility_logs.values() if log.get('is_test_circle', False))

    print(f"🔍 ELIGIBILITY SUMMARY:")
    print(f"   Total circles processed: {len(circle_eligibility_logs)}")
    print(f"   Eligible circles: {eligible_count}")
    print(f"   Small circles: {small_count}")
    print(f"   Test circles: {test_count}")

    # Verify all expected circles have eligibility logs
    missing_circles = [c_id for c_id in existing_circles.keys() if c_id not in circle_eligibility_logs]
    if missing_circles:
        print(f"⚠️ WARNING: {len(missing_circles)} circles are missing eligibility logs")
        print(f"   Missing circle IDs: {missing_circles[:5]}{'...' if len(missing_circles) > 5 else ''}")
    else:
        print(f"✅ All {len(existing_circles)} circles have eligibility logs")

    # Identify viable circles for optimization
    viable_circles = {circle_id: circle_data for circle_id, circle_data in existing_circles.items() 
                     if circle_data.get('max_additions', 0) > 0}

    # ENHANCED VIABLE CIRCLE DETECTION: List all circles with capacity
    print(f"\n🔍 VIABLE CIRCLES DETECTION:")
    print(f"  Found {len(viable_circles)} viable circles with max_additions > 0")

    # Verify circle viability more thoroughly
    if viable_circles:
        print(f"  Viable circle IDs: {list(viable_circles.keys())}")
    else:
        print(f"  ⚠️ WARNING: No viable circles found with max_additions > 0!")
        print(f"  This will prevent any circles from being used in optimization")

        # Look for small circles that should be eligible regardless of preference
        small_circles_to_promote = {circle_id: circle_data for circle_id, circle_data in existing_circles.items()
                                  if len(circle_data.get('members', [])) < 5}

        if small_circles_to_promote:
            print(f"  🔍 Found {len(small_circles_to_promote)} small circles that should receive new members")
            print(f"  Small circle IDs: {list(small_circles_to_promote.keys())}")
            print(f"  ✅ CRITICAL FIX: Adding these small circles to viable circles regardless of preference")

            # Add small circles to viable circles with a reasonable max_additions
            for small_id, small_data in small_circles_to_promote.items():
                if small_data.get('max_additions', 0) == 0:
                    small_data['max_additions'] = 5 - len(small_data.get('members', []))  # Add enough to reach 5
                    small_data['preference_overridden'] = True
                    small_data['override_reason'] = 'Small circle needs to reach viable size'
                    small_data['original_preference'] = 'None'
                    existing_circles[small_id] = small_data
                    viable_circles[small_id] = small_data
                    print(f"    Added {small_id} with max_additions={small_data['max_additions']}")
                else:
                    print(f"    {small_id} already has max_additions={small_data.get('max_additions', 0)}")

    # REGION MAPPING VERIFICATION
    print("\n🔍 REGION MAPPING VERIFICATION:")
    print(f"  Current region being processed: {region}")
    print(f"  Normalized region name: {normalize_region_name(region)}")

    # Add extensive debug for region matching
    if debug_mode:
        print(f"\n📋 VIABLE CIRCLES DETAILED DEBUG:")
        all_circles_count = len(existing_circles)
        capacity_circles_count = sum(1 for c in existing_circles.values() if c.get('max_additions', 0) > 0)

        # Print detailed info for all existing circles
        print(f"\nALL EXISTING CIRCLES ({all_circles_count}):")
        for circle_id, circle_data in existing_circles.items():
            max_additions = circle_data.get('max_additions', 0)
            member_count = len(circle_data.get('members', []))
            subregion = circle_data.get('subregion', 'Unknown')
            meeting_time = circle_data.get('meeting_time', 'Unknown')
            region = circle_data.get('region', 'Unknown')
            is_viable = max_additions > 0

            print(f"  Circle {circle_id}:")
            print(f"    Region: {region}")
            print(f"    Subregion: {subregion}")
            print(f"    Meeting time: {meeting_time}")
            print(f"    Current members: {member_count}")
            print(f"    Max additions: {max_additions}")
            print(f"    Is viable: {'✅ Yes' if is_viable else '❌ No'}")

        # Show viable circles summary
        print(f"\nVIABLE CIRCLES SUMMARY:")
        print(f"  {capacity_circles_count} of {all_circles_count} circles have capacity for new members")
        print(f"  {len(viable_circles)} circles will be used in optimization")

        # Print all circles with capacity
        if capacity_circles_count > 0:
            print(f"  Circles with capacity:")
            for circle_id, circle in existing_circles.items():
                if circle.get('max_additions', 0) > 0:
                    print(f"    {circle_id}: region='{circle.get('region', 'unknown')}', max_additions={circle.get('max_additions', 0)}")

    # Track circles at capacity (configurable maximum members)
    import streamlit as st
    max_circle_size = st.session_state.get('max_circle_size', 8) if 'st' in globals() else 8
    for circle in circles:
        if circle.get('member_count', 0) >= max_circle_size:
            optimization_context['full_circles'].append(circle.get('circle_id'))

    # Track circles needing hosts
    for circle in circles:
        if (circle.get('always_hosts', 0) == 0 and 
            circle.get('sometimes_hosts', 0) < 2 and
            circle.get('circle_id', '').startswith('IP-')):
            optimization_context['circles_needing_hosts'].append(circle)

    # CRITICAL FIX: Manually extract CURRENT-CONTINUING members and their circle IDs
    # before handling cases where no preferences exist
    current_continuing_members = {}
    # Track count of CURRENT-CONTINUING members
    cc_member_count = 0
    missing_circle_count = 0

    # Find and capture all CURRENT-CONTINUING members
    for _, participant in region_df.iterrows():
        # Check if this is a CURRENT-CONTINUING participant
        if participant.get('Status') == 'CURRENT-CONTINUING':
            cc_member_count += 1

            # Import utility to find circle ID
            from modules.optimizer_fixes import find_current_circle_id

            # Get their current circle ID using all possible methods
            current_circle = find_current_circle_id(participant)

            # If we found a valid circle ID, save it for special handling
            if current_circle:
                participant_id = participant.get('Encoded ID')
                if participant_id:
                    current_continuing_members[participant_id] = current_circle
            else:
                missing_circle_count += 1

    # Log what we found
    print(f"\n🔍 FOUND {cc_member_count} CURRENT-CONTINUING MEMBERS IN REGION {region}")
    print(f"  Of those, {len(current_continuing_members)} have valid circle IDs")
    print(f"  {missing_circle_count} members have missing circle IDs")

    # Handle case where no preferences exist for NEW participants
    # But still allow CURRENT-CONTINUING members to match with their circles
    if not subregions or not time_slots:
        print(f"\n🚨 CRITICAL FIX: Region {region} has no valid preferences, but we'll still match CURRENT-CONTINUING members")

        results = []
        unmatched = []
        matched_circles = []

        # Process each participant
        for _, participant in region_df.iterrows():
            participant_dict = participant.to_dict()
            participant_id = participant.get('Encoded ID')
            is_continuing = participant.get('Status') == 'CURRENT-CONTINUING'

            # Check if this is a CURRENT-CONTINUING participant with a known circle
            if is_continuing and participant_id in current_continuing_members:
                # Get their assigned circle
                assigned_circle = current_continuing_members[participant_id]

                # Mark as matched to their current circle
                participant_dict['proposed_NEW_circles_id'] = assigned_circle
                participant_dict['location_score'] = 100  # Max score
                participant_dict['time_score'] = 100  # Max score  
                participant_dict['total_score'] = 200  # Max total score
                participant_dict['unmatched_reason'] = "FIXED: Manually assigned to continuing circle"

                # Record this assignment for return values
                results.append(participant_dict)
                matched_circles.append({
                    'circle_id': assigned_circle,
                    'member_count': 1,  # Just counting this member for now
                    'members': [participant_dict]
                })

                print(f"  ✅ CURRENT-CONTINUING member {participant_id} matched to circle {assigned_circle}")

            else:
                # This is either a NEW participant or CURRENT-CONTINUING without a known circle
                # Mark as unmatched with appropriate reason
                participant_dict['proposed_NEW_circles_id'] = "UNMATCHED"

                # Set scores to 0 for unmatched participants
                participant_dict['location_score'] = 0
                participant_dict['time_score'] = 0
                participant_dict['total_score'] = 0

                # Set the reason based on what's missing
                if not subregions and not time_slots:
                    participant_dict['unmatched_reason'] = "No valid preferences found (both location and time)"
                elif not subregions:
                    participant_dict['unmatched_reason'] = "No valid location preferences found"
                elif not time_slots:
                    participant_dict['unmatched_reason'] = "No valid time preferences found"

                # Record as unmatched
                results.append(participant_dict)
                unmatched.append(participant_dict)

        # Return both matched and unmatched participants
        return results, matched_circles, unmatched, {}, circle_eligibility_logs

    # ***************************************************************
    # STEP 1: STRUCTURE VARIABLES AROUND REAL AND HYPOTHETICAL CIRCLES
    # ***************************************************************

    # Prepare existing circle IDs (real IDs like IP-BOS-02)
    existing_circle_ids = list(viable_circles.keys())

    # Create synthetic IDs for potential new circles based on subregion and time
    # Include both existing circle patterns and new circle options from preferences
    new_circle_candidates = [(subregion, time_slot) for subregion in subregions for time_slot in time_slots]

    # Add new circle options generated from participant preferences (greenfield scenarios)
    if new_circle_options:
        new_circle_candidates.extend(new_circle_options)
        if debug_mode:
            print(f"🔧 Enhanced new circle candidates: {len(new_circle_candidates)} total ({len(new_circle_options)} from preferences)")

    # Generate synthetic circle IDs for potential new circles
    new_circle_ids = []
    new_circle_metadata = {}  # Map IDs to their subregion and time

    # Import region code mapping utilities
    from utils.normalization import get_region_code, get_region_code_with_subregion

    # Step 1: Group potential circles by region for sequential numbering
    regions_and_times = {}

    # Get the standardized region code for the current region
    is_virtual = "Virtual" in region if region is not None else False

    # Initialize counter for this region - always start from 1
    counter = 1

    # Determine format based on whether it's virtual or in-person
    format_prefix = "VO" if is_virtual else "IP"

    if debug_mode:
        print(f"Creating new circle IDs for region {region} (is_virtual: {is_virtual})")

    # For each subregion and time slot combination
    for subregion, time_slot in new_circle_candidates:
        # Format the counter as a 2-digit number (01, 02, etc.)
        circle_num = str(counter).zfill(2)

        # Get the appropriate region code with enhanced validation
        if is_virtual and subregion:
            # For virtual circles, use the region code with timezone from subregion
            region_code = get_region_code_with_subregion(region, subregion, is_virtual=True)

            # CRITICAL FIX: Ensure we never use 'Invalid' or 'Unknown' for virtual circles
            if region_code in ['Invalid', 'Unknown', 'UNKNOWN']:
                print(f"⚠️ WARNING: Invalid region code '{region_code}' for virtual circle, using fallback")
                # Use enhanced fallback based on region type
                if 'APAC' in region and 'EMEA' not in region:
                    region_code = 'AP-GMT'
                elif 'EMEA' in region and 'APAC' not in region:
                    region_code = 'EM-GMT'
                elif 'Americas' in region:
                    region_code = 'AM-GMT'
                else:
                    region_code = 'VO-GMT'
                print(f"  Applied fallback region code: {region_code}")

            if debug_mode:
                print(f"  Virtual circle with subregion {subregion}, using region_code: {region_code}")
        else:
            # For in-person circles, use the standard region code
            region_code = get_region_code(region)

            # CRITICAL FIX: Ensure we never use 'Invalid' or 'Unknown' for any circles
            if region_code in ['Invalid', 'Unknown', 'UNKNOWN']:
                print(f"⚠️ WARNING: Invalid region code '{region_code}' for in-person circle, using fallback")
                region_code = 'UNKNOWN'

        # CRITICAL FIX: Ensure format prefix matches the region code for virtual circles
        if is_virtual and not format_prefix == "VO":
            print(f"⚠️ CRITICAL FIX: Correcting format prefix from {format_prefix} to VO for virtual circle")
            format_prefix = "VO"

        # Generate a unique ID for this potential new circle using the correct format
        circle_id = f"{format_prefix}-{region_code}-NEW-{circle_num}"

        if debug_mode:
            print(f"  Created new circle ID: {circle_id} for subregion: {subregion}, time: {time_slot}")

        # Increment the counter for the next circle in this region
        counter += 1

        new_circle_ids.append(circle_id)
        new_circle_metadata[circle_id] = {
            'subregion': subregion,
            'meeting_time': time_slot,
            'region': region
        }

    # Combine all circle IDs (existing + potential new)
    all_circle_ids = existing_circle_ids + new_circle_ids

    # Create a mapping from circle ID to its metadata
    circle_metadata = {}

    # Add existing circle metadata
    for circle_id, circle_data in viable_circles.items():
        circle_metadata[circle_id] = {
            'subregion': circle_data.get('subregion', ''),
            'meeting_time': circle_data.get('meeting_time', ''),
            'region': circle_data.get('region', ''),
            'max_additions': circle_data.get('max_additions', 0),
            'is_existing': True,
            'current_members': len(circle_data.get('members', [])),
            'circle_data': circle_data  # Keep the original data for reference
        }

    # Add new circle metadata
    for circle_id in new_circle_ids:
        circle_metadata[circle_id] = {
            'subregion': new_circle_metadata[circle_id]['subregion'],
            'meeting_time': new_circle_metadata[circle_id]['meeting_time'],
            'region': new_circle_metadata[circle_id]['region'],
            'max_additions': max_circle_size,  # New circles can have up to configurable maximum members
            'is_existing': False,
            'current_members': 0
        }

    if debug_mode:
        print(f"\n🔄 REFACTORED CIRCLE SETUP:")
        print(f"  Existing circles: {len(existing_circle_ids)}")
        print(f"  Potential new circles: {len(new_circle_ids)}")
        print(f"  Total circles: {len(all_circle_ids)}")

        # Show some example circles
        if existing_circle_ids:
            for circle_id in existing_circle_ids[:3]:
                meta = circle_metadata[circle_id]
                print(f"  Example existing circle: {circle_id}")
                print(f"    Subregion: {meta['subregion']}")
                print(f"    Meeting time: {meta['meeting_time']}")
                print(f"    Max additions: {meta['max_additions']}")
                print(f"    Current members: {meta['current_members']}")

        if new_circle_ids:
            for circle_id in new_circle_ids[:3]:
                meta = circle_metadata[circle_id]
                print(f"  Example potential new circle: {circle_id}")
                print(f"    Subregion: {meta['subregion']}")
                print(f"    Meeting time: {meta['meeting_time']}")

    # ***************************************************************
    # DIAGNOSTIC STEP: TRACK CURRENT-CONTINUING MEMBERS
    # ***************************************************************

    # Track CURRENT-CONTINUING members through our diagnostic tools
    print("\n🔍 DIAGNOSTIC: Tracking CURRENT-CONTINUING members")
    continuing_debug_info = track_current_continuing_status(region_df)
    print(f"  Identified {continuing_debug_info['total_continuing_members']} CURRENT-CONTINUING members in {region} region")
    print(f"  Success rates for column detection methods:")
    print(f"    - Standard method: {continuing_debug_info['standard_method_success_rate']:.2%}")
    print(f"    - Hybrid method: {continuing_debug_info['hybrid_method_success_rate']:.2%}")
    print(f"    - Aggressive method: {continuing_debug_info['aggressive_method_success_rate']:.2%}")
    print(f"    - Any method: {continuing_debug_info['any_method_success_rate']:.2%}")

    # ***************************************************************
    # STEP 1.5: CRITICAL FIXES FOR CURRENT-CONTINUING MEMBERS AND OPTIMIZE MODE
    # ***************************************************************

    # CRITICAL FIX 1: Apply capacity optimization for continuing circles in "optimize" mode
    print("\n🚨 APPLYING CRITICAL FIXES FOR CIRCLE CAPACITY IN OPTIMIZE MODE")
    # Update viable_circles with optimized capacity values
    viable_circles = optimize_circle_capacity(viable_circles, existing_circle_handling, min_circle_size)

    # CRITICAL FIX 2: Pre-process all CURRENT-CONTINUING members
    print("\n🚨 PRE-PROCESSING CURRENT-CONTINUING MEMBERS")

    # Get IDs of all participants in this region
    all_participant_ids = region_df['Encoded ID'].tolist()

    # Pre-assign CURRENT-CONTINUING members to their existing circles
    preassigned_circles, problem_participants = preprocess_continuing_members(
        region_df, 
        existing_circle_ids
    )

    if preassigned_circles:
        print(f"✅ Successfully pre-assigned {len(preassigned_circles)} CURRENT-CONTINUING members to their circles")
    else:
        print("⚠️ No CURRENT-CONTINUING members were pre-assigned")

    if problem_participants:
        print(f"⚠️ Found {len(problem_participants)} CURRENT-CONTINUING members with problems:")
        for p in problem_participants[:5]:  # Show first 5 for brevity
            print(f"  - Participant {p['participant_id']}: {p['reason']}")

        if len(problem_participants) > 5:
            print(f"  ... and {len(problem_participants) - 5} more problematic participants")

    # ***************************************************************
    # STEP 2: DEFINE DECISION VARIABLES
    # ***************************************************************

    # Set up the optimization problem
    prob = pulp.LpProblem(f"CircleMatching_{region}", pulp.LpMaximize)

    # Create decision variables:
    # x[p_id, c_id] = 1 if participant p_id is assigned to circle c_id
    x = {}

    # Track which variables were created for verification
    created_vars = []

    # Initialize Seattle debug logs in Streamlit session state if needed
    import streamlit as st
    if 'seattle_debug_logs' not in st.session_state:
        st.session_state.seattle_debug_logs = []

    # Just use the participants from the remaining dataframe without special cases
    participants = remaining_df['Encoded ID'].tolist()

    # Log participants being processed in this region
    print(f"\n🔍 Processing {len(participants)} participants in region {region}")

    # Define test participants for debug logging only (but no special handling)
    # Use Seattle-specific test case
    test_participants = ['99999000001']  # Seattle test ID
    test_circles = ['IP-SEA-01']  # Seattle test circle

    # Log which test participants are in this region (for debugging only)
    for test_id in test_participants:
        if test_id in participants:
            test_row = remaining_df[remaining_df['Encoded ID'] == test_id]
            if not test_row.empty:
                first_row = test_row.iloc[0]
                print(f"DEBUG: Test participant {test_id} found in region {region}")
                print(f"  Status: {first_row.get('Status', 'Unknown')}")
                print(f"  Region: {first_row.get('Derived_Region', first_row.get('Current_Region', 'Unknown'))}")
                # For Seattle test participant, add to debug logs
                if region == "Seattle" and test_id == "99999000001":
                    st.session_state.seattle_debug_logs.append(f"Found Seattle test participant {test_id} in region {region}")

    # Special logging if this is Seattle region
    if region == "Seattle":
        print(f"🔍 Processing Seattle region: {region}")
        st.session_state.seattle_debug_logs.append(f"Processing Seattle region with '{existing_circle_handling}' mode")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ***************************************************************
    print("\n🚨 CRITICAL FIX: Pre-assigning CURRENT-CONTINUING members to their current circles")

    # Find the column containing current circle IDs
    current_col = None
    potential_columns = ['current_circles_id', 'Current_Circle_ID', 'Current Circle ID', 'Current Circle', 'Current_Circle']

    # Print available columns for debugging
    print(f"  Available columns: {region_df.columns.tolist()}")

    # First try direct matches
    for col in potential_columns:
        if col in region_df.columns:
            current_col = col
            print(f"  ✅ Found exact column match: '{col}'")
            break

    # If not found, try case-insensitive matching
    if current_col is None:
        for col in region_df.columns:
            if any(potential.lower() in col.lower() for potential in potential_columns):
                current_col = col
                print(f"  ✅ Found case-insensitive match: '{col}'")
                break

    # If still not found, try to search for any column with 'circle' and 'id' or 'current'
    if current_col is None:
        for col in region_df.columns:
            if 'circle' in col.lower() and ('id' in col.lower() or 'current' in col.lower()):
                current_col = col
                print(f"  ✅ Found fuzzy match: '{col}'")
                break

    # Print the result of our search
    if current_col:
        print(f"  Using column '{current_col}' for current circle IDs")
        # Show some sample values
        non_null_values = region_df[region_df[current_col].notna()][current_col].unique()
        print(f"  Sample values: {non_null_values[:5].tolist() if len(non_null_values) > 0 else 'No non-null values'}")
    else:
        print("  ⚠️ No column found for current circle IDs - this will affect CURRENT-CONTINUING assignments")

    # Initialize pre-assignment tracking
    pre_assigned_participants = {}  # Maps participant IDs to their assigned circle
    pre_assigned_circles = {}  # Maps circles to list of pre-assigned participants
    participants_to_remove = []  # Track participants to remove from optimization

    if current_col is not None:
        # Find all CURRENT-CONTINUING participants with circle IDs
        continuing_df = region_df[(region_df['Status'] == 'CURRENT-CONTINUING') & region_df[current_col].notna()]

        print(f"  Found {len(continuing_df)} CURRENT-CONTINUING participants with non-null circle IDs")

        if not continuing_df.empty:
            # Process each CURRENT-CONTINUING participant
            for idx, row in continuing_df.iterrows():
                p_id = row['Encoded ID']
                circle_id = str(row[current_col]).strip()

                # Skip empty or NaN circle IDs (shouldn't happen due to filter above, but just in case)
                if not circle_id or pd.isna(circle_id):
                    continue

                # Check if this participant is in our participants list (they should be)
                if p_id in participants:
                    # Record the pre-assignment
                    pre_assigned_participants[p_id] = circle_id

                    # Add to circle's pre-assigned list
                    if circle_id not in pre_assigned_circles:
                        pre_assigned_circles[circle_id] = []
                    pre_assigned_circles[circle_id].append(p_id)

                    # Mark to remove from participants list to avoid double assignment
                    participants_to_remove.append(p_id)
                else:
                    print(f"  ⚠️ CURRENT-CONTINUING participant {p_id} not found in participants list")

            # Remove pre-assigned participants from the optimization pool
            for p_id in participants_to_remove:
                if p_id in participants:
                    participants.remove(p_id)

            print(f"  Pre-assigned {len(pre_assigned_participants)} CURRENT-CONTINUING participants to {len(pre_assigned_circles)} circles")

            # Update circle capacities after pre-assignment
            for circle_id, assigned_participants in pre_assigned_circles.items():
                if circle_id in circle_metadata:
                    # Reduce max_additions by the number of pre-assigned participants
                    current_max = circle_metadata[circle_id]['max_additions']
                    new_max = max(0, current_max - len(assigned_participants))
                    circle_metadata[circle_id]['max_additions'] = new_max

                    print(f"  Updated capacity for circle {circle_id}: {current_max} → {new_max} remaining slots")
    else:
        print("  ⚠️ CRITICAL ERROR: Could not find current circle ID column in the dataframe")

    print(f"  Continuing optimization with {len(participants)} remaining participants (mainly NEW)")
    # End of CURRENT-CONTINUING pre-assignment

    # Create variables for all participant-circle pairs
    for p_id in participants:
        # Get row data from dataframe (with defensive coding)
        matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
        if matching_rows.empty:
            print(f"⚠️ Warning: Participant {p_id} not found in region dataframe")
            # Skip this participant to avoid errors
            continue

        # Get participant data
        p_row = matching_rows.iloc[0]

        # For debug logging only
        is_houston_participant = any('Houston' in str(loc) if isinstance(loc, str) else False for loc in [
            p_row.get('first_choice_location'), 
            p_row.get('second_choice_location'), 
            p_row.get('third_choice_location')
        ])

        for c_id in all_circle_ids:
            # For debug logging only
            is_test_case = (p_id in test_participants and c_id in test_circles)

            # Create all variables regardless of compatibility - constraints will handle restrictions
            x[(p_id, c_id)] = pulp.LpVariable(f"x_{p_id}_{c_id}", cat=pulp.LpBinary)
            created_vars.append((p_id, c_id))

            # Special debug for Houston-related variables
            is_houston_circle = 'HOU' in c_id if c_id is not None else False
            if (is_houston_participant or is_houston_circle or is_test_case):
                print(f"✅ Created LP variable for {p_id} ↔ {c_id}")
                # Focus on Seattle test cases
                if is_test_case and region == "Seattle":
                    st.session_state.seattle_debug_logs.append(f"VERIFIED CREATION of test variable x[{p_id}, {c_id}]")

    # Participants list already created above

    # Create dictionary to track compatibility between participants and circles
    compatibility = {}
    participant_compatible_circles = {}
    for p_id in participants:
        participant_compatible_circles[p_id] = []

    # CRITICAL FIX: Apply compatibility fixes from preassigned circles from earlier steps
    for p_id, circle_id in preassigned_circles.items():
        # Force compatibility between this participant and their current circle
        compatibility[(p_id, circle_id)] = 1
        print(f"✅ Forced compatibility between {p_id} and {circle_id} for CURRENT-CONTINUING member")

    if debug_mode:
        print(f"\n🔢 Created {len(created_vars)} LP variables for {len(participants)} participants and {len(all_circle_ids)} circles")

        # Verify crucial variables exist for Houston circles
        for p_id in participants:
            for c_id in existing_circle_ids:
                if 'HOU' in c_id:
                    if (p_id, c_id) in x:
                        print(f"✅ Confirmed LP variable exists for pair: {p_id} ↔ {c_id}")
                    else:
                        print(f"❌ ERROR: No LP variable for Houston pair: {p_id} ↔ {c_id}")

        # Special debug for test participants
        for p_id in test_participants:
            if p_id in participants:
                for c_id in test_circles:
                    if c_id in all_circle_ids:
                        if (p_id, c_id) in x:
                            print(f"✅ Confirmed LP variable exists for test pair: {p_id} ↔ {c_id}")
                        else:
                            print(f"❌ ERROR: No LP variable for test pair: {p_id} ↔ {c_id}")

    # Create binary variables for circle activation (only needed for new circles)
    y = {}
    for c_id in new_circle_ids:
        y[c_id] = pulp.LpVariable(f"y_{c_id}", cat=pulp.LpBinary)

    # Track test participants without special handling
    # Use Seattle-specific test case
    seattle_test_id = '99999000001'

    # For Seattle regions, add special diagnostic info
    if region == "Seattle":
        st.session_state.seattle_debug_logs.append(f"Processing Seattle with '{existing_circle_handling}' circle handling mode")
        st.session_state.seattle_debug_logs.append(f"Will track Seattle test participant {seattle_test_id} in normal matching process")

    # ***************************************************************
    # CRITICAL FIX: PRE-ASSIGN CURRENT-CONTINUING MEMBERS TO THEIR CURRENT CIRCLES
    # ****************************************************************
    # THE FIX: In the code below, for MOVING INTO Region participants, we ensure that their `proposed_NEW_Subregion` is set to the `subregion` of the circle they are assigned to, rather than keeping their `Current_Subregion`. This is crucial because these participants are actively moving into a new region/subregion, and their `proposed_NEW_Subregion` should reflect their destination, not their origin.
    # This logic is implemented within the loop where we process assignments.
    # We explicitly assign `subregion` (from the circle's metadata) to `proposed_NEW_Subregion`.
    # This fixes the issue where `proposed_NEW_Subregion` was incorrectly showing the participant's `Current_Subregion` after assignment.
    # The original code snippet had a comment that said `participant_dict['proposed_NEW_Subregion'] = subregion` but it was not there.
    # Let's add it and ensure it is used.
    # The same logic applies to `proposed_NEW_DayTime`.
    # The original logic was:
    #                 participant_dict['proposed_NEW_circles_id'] = circle_id
    #                 participant_dict['proposed_NEW_Subregion'] = subregion
    #                 participant_dict['proposed_NEW_DayTime'] = meeting_time
    #
    # The corrected logic should be:
    #                 participant_dict['proposed_NEW_circles_id'] = circle_id
    #                 # CRITICAL FIX: Always use circle's subregion, not participant's preference
    #                 # This is especially important for MOVING INTO Region participants
    #                 participant_dict['proposed_NEW_Subregion'] = subregion
    #                 participant_dict['proposed_NEW_DayTime'] = meeting_time
    #
    # The change is to uncomment and ensure `proposed_NEW_Subregion` is assigned from `subregion`.
    # ***************************************************************

    # For each participant, determine their assignment and associated metadata
    for p_id in participants:
        # Check if this participant was assigned to any circle
        if p_id in circle_assignments:
            c_id = circle_assignments[p_id]
            # Get the metadata for the assigned circle
            # Check if c_id is valid and present in circle_metadata before accessing
            if c_id in circle_metadata:
                meta = circle_metadata[c_id]
                subregion = meta.get('subregion', 'Unknown')
                meeting_time = meta.get('meeting_time', 'Unknown')

                # Retrieve participant data from the original region_df
                matching_rows = remaining_df[remaining_df['Encoded ID'] == p_id]
                if not matching_rows.empty:
                    participant_dict = matching_rows.iloc[0].to_dict()

                    # Set the participant's assignment data
                    participant_dict['proposed_NEW_circles_id'] = c_id
                    # CRITICAL FIX: Always use circle's subregion, not participant's preference
                    # This is especially important for MOVING INTO Region participants
                    participant_dict['proposed_NEW_Subregion'] = subregion
                    participant_dict['proposed_NEW_DayTime'] = meeting_time

                    # Calculate preference scores
                    loc_score = 0
                    time_score = 0

                    # Location score
                    if participant_dict.get('first_choice_location') == subregion:
                        loc_score = 30
                    elif participant.get('second_choice_location') == subregion:
                        loc_score = 20
                    elif participant.get('third_choice_location') == subregion:
                        loc_score = 10

                    # Time score
                    first_choice = participant_dict.get('first_choice_time', '')
                    second_choice = participant_dict.get('second_choice_time', '')
                    third_choice = participant_dict.get('third_choice_time', '')

                    # Define if this is a special test case
                    is_test_case = (p_id == '73177784103' and c_id == 'IP-SIN-01') or (p_id == '50625303450' and c_id == 'IP-LON-04')

                    # Check first choice using is_time_compatible for consistent handling of "Varies"
                    if is_time_compatible(first_choice, time_slot, is_important=is_test_case):
                        time_score = 30
                    # Check second choice
                    elif is_time_compatible(second_choice, time_slot, is_important=is_test_case):
                        time_score = 20
                    # Check third choice
                    elif is_time_compatible(third_choice, time_slot, is_important=is_test_case):
                        time_score = 10

                    participant_dict['location_score'] = loc_score
                    participant_dict['time_score'] = time_score
                    participant_dict['total_score'] = loc_score + time_score

                    # Add to results list
                    results.append(participant_dict)
                else:
                    print(f"⚠️ Participant {p_id} assigned to circle {c_id} but not found in remaining_df")
            else:
                # This participant was not assigned to any circle (i.e., unmatched)
                # This case should be handled later when building the final results list
                pass
        else:
            # This participant was not assigned to any circle (unmatched)
            # This case should be handled later when building the final results list
            pass

    # Handle unmatched participants separately to ensure they are included in results
    # Build a list of all participants that are still unmatched after the optimization
    all_assigned_participants = set(circle_assignments.keys())
    unmatched_participants_final = []

    for _, participant in region_df.iterrows():
        p_id = participant['Encoded ID']
        if p_id not in all_assigned_participants:
            participant_dict = participant.to_dict()
            participant_dict['proposed_NEW_circles_id'] = "UNMATCHED"
            participant_dict['location_score'] = 0
            participant_dict['time_score'] = 0
            participant_dict['total_score'] = 0

            # Use the context object and the determine_unmatched_reason function to get the reason
            # Build the context object with all necessary data
            detailed_context = {
                'existing_circles': optimization_context.get('existing_circles', []),
                'participant_compatible_options': optimization_context.get('participant_compatible_options', {}),
                'region_participant_count': globals().get('region_participant_count', {}),
                'debug_mode': debug_mode,
            }
            participant_dict['unmatched_reason'] = determine_unmatched_reason(participant, detailed_context)

            unmatched_participants_final.append(participant_dict)
            unmatched.append(participant_dict) # Add to the main unmatched list as well

    # Combine results: matched participants first, then unmatched
    final_results = results + unmatched_participants_final

    # CRITICAL FIX: Ensure that CURRENT-CONTINUING members are correctly assigned to their existing circles
    # even if they were not explicitly assigned during the optimization.
    # This is a fallback mechanism to ensure continuity.
    # It also ensures that if the optimization failed to assign them, they are assigned here.
    print(f"\n🚨 CRITICAL FIX: Final check for CURRENT-CONTINUING members in {region} region")

    # Initialize counts
    continuing_matched_count = 0
    continuing_unmatched_count = 0

    # Iterate through all participants in the region_df
    for _, participant in region_df.iterrows():
        p_id = participant['Encoded ID']
        status = participant.get('Status', '')

        # Check if this is a CURRENT-CONTINUING member
        if status in ['CURRENT-CONTINUING', 'Current-CONTINUING']:
            # Check if they were already assigned by the optimizer
            is_assigned = False
            for result in final_results:
                if result.get('Encoded ID') == p_id and result.get('proposed_NEW_circles_id') != "UNMATCHED":
                    is_assigned = True
                    continuing_matched_count += 1
                    break

            # If not assigned by the optimizer, try to find their current circle and assign them manually
            if not is_assigned:
                continuing_unmatched_count += 1
                print(f"  ⚠️ CURRENT-CONTINUING member {p_id} was not assigned by optimizer.")

                # Try to find their current circle ID using the helper function
                current_circle_id = find_current_circle_id(participant)

                # If a circle ID was found and it's a valid existing circle
                if current_circle_id and current_circle_id in existing_circle_ids:
                    print(f"  ✅ Found current circle '{current_circle_id}'. Manually assigning {p_id}.")

                    # Find the metadata for this circle
                    circle_meta = None
                    # Need to check both 'viable_circles' and 'existing_circles' as 'viable_circles' is derived from 'existing_circles'
                    if current_circle_id in viable_circles:
                        circle_meta = viable_circles[current_circle_id]
                    elif current_circle_id in existing_circles:
                        circle_meta = existing_circles[current_circle_id]
                    else:
                        print(f"  ⚠️ CRITICAL WARNING: Circle {current_circle_id} found for {p_id} but not in existing_circles/viable_circles.")
                        # We'll proceed with assigning it anyway, but this might indicate an issue.

                    if circle_meta:
                        subregion = circle_meta.get('subregion', 'Unknown')
                        meeting_time = circle_meta.get('meeting_time', 'Unknown')

                        # Create a dictionary for this assignment
                        assigned_participant = participant.to_dict()
                        assigned_participant['proposed_NEW_circles_id'] = current_circle_id
                        assigned_participant['proposed_NEW_Subregion'] = subregion
                        assigned_participant['proposed_NEW_DayTime'] = meeting_time
                        assigned_participant['unmatched_reason'] = "FIXED: Manually assigned to continuing circle"
                        assigned_participant['location_score'] = 3  # Default high score
                        assigned_participant['time_score'] = 3
                        assigned_participant['total_score'] = 6

                        # Add to results list and remove from unmatched list
                        final_results.append(assigned_participant)
                        # Remove from unmatched if it exists there
                        unmatched = [p for p in unmatched if p.get('Encoded ID') != p_id]
                else:
                    print(f"  ❌ Failed to find a valid current circle for {p_id}.")
                    # If no valid circle, ensure it's marked as unmatched
                    if participant_dict.get('proposed_NEW_circles_id') != "UNMATCHED":
                        participant_dict = participant.to_dict()
                        participant_dict['proposed_NEW_circles_id'] = "UNMATCHED"
                        participant_dict['unmatched_reason'] = 'ERROR: CURRENT-CONTINUING with no valid circle ID found'
                        participant_dict['location_score'] = 0
                        participant_dict['time_score'] = 0
                        participant_dict['total_score'] = 0
                        # Add this to results if not already there (which it shouldn't be if it was unmatched)
                        already_in_results = any(res.get('Encoded ID') == p_id for res in final_results)
                        if not already_in_results:
                            final_results.append(participant_dict)

    print(f"  CURRENT-CONTINUING members: {continuing_matched_count} matched, {continuing_unmatched_count} previously unmatched (now processed).")


    # ***************************************************************
    # DIAGNOSTIC STEP: TRACK FINAL MATCHING OUTCOMES
    # ***************************************************************
    print("\n🔍 DIAGNOSTIC: Tracking final matching outcomes and circle assignments")

    # Re-run track_matching_outcomes with the FINAL results
    final_matching_outcomes = track_matching_outcomes(continuing_debug_info, final_results, unmatched)
    print(f"  Final Match rate: {final_matching_outcomes['match_rate']:.2%}")
    print(f"  Final Correct match rate: {final_matching_outcomes['correct_match_rate']:.2%}")

    # Add circle assignments to the final results dictionary
    # This helps in reconstructing the full circle data for UI and reporting
    # We will consolidate this into the `circles` list which already contains circle info
    
    # For each circle, identify its members from the final results
    circle_assignments_map = {} # Map circle_id -> list of participant dictionaries
    for result in final_results:
        circle_id = result.get('proposed_NEW_circles_id')
        if circle_id and circle_id != "UNMATCHED":
            if circle_id not in circle_assignments_map:
                circle_assignments_map[circle_id] = []
            circle_assignments_map[circle_id].append(result)

    # Update the 'circles' list with member details and correct counts
    final_circles_list = []
    processed_circle_ids_for_final = set() # To avoid duplicate circle entries

    # Add existing viable circles first, with updated member counts and new member info
    for circle_id, circle_data in viable_circles.items():
        if circle_id not in processed_circle_ids_for_final:
            updated_circle = circle_data.copy()
            members_in_results = circle_assignments_map.get(circle_id, [])
            updated_circle['members'] = [m['Encoded ID'] for m in members_in_results]
            updated_circle['member_count'] = len(updated_circle['members'])
            updated_circle['new_members'] = len([m for m in members_in_results if m.get('Status') == 'NEW'])
            final_circles_list.append(updated_circle)
            processed_circle_ids_for_final.add(circle_id)

    # Add new circles created by the optimization
    for circle in circles: # Original 'circles' list contains newly formed circles
        if circle.get('circle_id') not in processed_circle_ids_for_final:
            members_in_results = circle_assignments_map.get(circle['circle_id'], [])
            circle['members'] = [m['Encoded ID'] for m in members_in_results]
            circle['member_count'] = len(circle['members'])
            circle['new_members'] = len(members_in_results)
            final_circles_list.append(circle)
            processed_circle_ids_for_final.add(circle['circle_id'])
        elif debug_mode:
            print(f"Skipped adding duplicate circle {circle['circle_id']} to final_circles_list")

    # CRITICAL FIX: Add FINAL circle eligibility logs to session state
    # This ensures the UI component has the most up-to-date eligibility info
    print(f"\n🚨 CRITICAL FIX: Updating session state with FINAL circle eligibility logs for {region}")
    # Use the `final_logs` which is already a copy and contains the final state
    if 'circle_eligibility_logs' in st.session_state:
        st.session_state.circle_eligibility_logs = final_logs
        print(f"✅ Session state updated with {len(final_logs)} final eligibility logs.")
    else:
        st.session_state.circle_eligibility_logs = final_logs
        print(f"✅ Created and updated session state with {len(final_logs)} final eligibility logs.")

    # FINAL VERIFICATION: Ensure the logs contains valid entries
    print(f"\n🚨 FINAL LOG COUNT CHECK FOR {region}: {len(final_logs)} entries")
    # Log each entry creation for better debugging
    print(f"\n🔴 CRITICAL LOG CHECK: Circle eligibility for region {region}")
    print(f"CREATED {len(final_logs)} LOGS - DETAILED REGISTRY:")

    # Show the exact contents of circle_eligibility_logs
    if final_logs:
        print(f"Circle IDs with eligibility logs: {list(final_logs.keys())}")
    else:
        print("❌ CRITICAL ERROR: No circle eligibility logs were created!")

    # CRITICAL FIX: Add ALL circles with capacity to debugging info, ensuring consistency
    print(f"\n🚨 CRITICAL FIX: Populating circle_capacity_debug with ALL circles having capacity")
    # Reset the dictionary to ensure fresh data
    st.session_state.circle_capacity_debug = {}
    circles_with_capacity_count = 0

    for circle_id in all_circle_ids:
        # Check if the circle has capacity (max_additions > 0)
        # We need to refer to the circle_metadata for max_additions
        if circle_id in circle_metadata:
            max_additions = circle_metadata[circle_id].get('max_additions', 0)
            if max_additions > 0:
                circles_with_capacity_count += 1
                # Get all relevant info from circle_metadata
                meta = circle_metadata[circle_id]
                st.session_state.circle_capacity_debug[circle_id] = {
                    'circle_id': circle_id,
                    'region': meta.get('region', 'Unknown'),
                    'subregion': meta.get('subregion', 'Unknown'),
                    'meeting_time': meta.get('meeting_time', 'Unknown'),
                    'current_members': meta.get('current_members', 0),
                    'max_additions': max_additions,
                    'viable': True,  # Mark as viable since it has capacity
                    'is_test_circle': circle_id in ['IP-SIN-01', 'IP-LON-04', 'IP-HOU-02', 'IP-SEA-01'], # Add Seattle test circle
                    'special_handling': circle_id in ['IP-SIN-01', 'IP-LON-04', 'IP-HOU-02', 'IP-SEA-01'] # Add Seattle test circle
                }
        else:
            print(f"  ⚠️ CRITICAL ERROR: Circle ID {circle_id} not found in circle_metadata. Skipping capacity check.")

    print(f"  Added {circles_with_capacity_count} circles with capacity to circle_capacity_debug")

    # If any Seattle circles are present, add extra debug logs
    if region == "Seattle" and any('SEA' in c_id for c_id in all_circle_ids):
        st.session_state.seattle_debug_logs.append(f"\nFINAL CAPACITY CHECK FOR SEATTLE CIRCLES:")
        for c_id in all_circle_ids:
            if 'SEA' in c_id:
                st.session_state.seattle_debug_logs.append(f"  Circle {c_id}:")
                if c_id in circle_metadata:
                    meta = circle_metadata[c_id]
                    st.session_state.seattle_debug_logs.append(f"    Max additions: {meta.get('max_additions', 0)}")
                    st.session_state.seattle_debug_logs.append(f"    Current members: {len(meta.get('members', []))}")
                    # Check eligibility from final_logs
                    if c_id in final_logs:
                        elig_info = final_logs[c_id]
                        st.session_state.seattle_debug_logs.append(f"    Eligibility: {'YES' if elig_info.get('is_eligible') else 'NO'}")
                        if 'reason' in elig_info:
                            st.session_state.seattle_debug_logs.append(f"    Reason: {elig_info['reason']}")
                    else:
                        st.session_state.seattle_debug_logs.append(f"    No eligibility info found for {c_id}")
                else:
                    st.session_state.seattle_debug_logs.append(f"    Metadata not found for {c_id}")

    # Return the processed results, circles, unmatched list, debug info, and final logs
    return final_results, final_circles_list, unmatched, circle_capacity_debug, final_logs

# East Bay debug function was removed to focus exclusively on Seattle test case