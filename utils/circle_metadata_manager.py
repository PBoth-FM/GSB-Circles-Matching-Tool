import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union, Optional, Set

class CircleMetadataManager:
    """
    Central manager for all circle metadata throughout the application.
    Provides a single source of truth for circle data.
    
    This class handles both continuing and new circles with consistent interfaces
    while providing specialized handling where needed based on circle type.
    """
    
    def __init__(self):
        self.circles = {}  # Dictionary keyed by circle_id
        self.results_df = None  # Reference to participant results DataFrame
        self._initialized = False
        self.logger = self._setup_logger()
        self.split_circles = {}  # Dictionary of split circle IDs with original circle ID as value
        self.original_circles = {}  # Dictionary of original circle IDs with list of split circle IDs as value
    
    def _setup_logger(self):
        """Setup a logger for the metadata manager"""
        logger = logging.getLogger('circle_metadata_manager')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    def initialize_from_optimizer(self, optimizer_circles, 
                               results_df, participant_manager=None) -> 'CircleMetadataManager':
        """
        Initialize circle metadata from optimizer results and store reference to results DataFrame
        
        Args:
            optimizer_circles: List of circle dictionaries from optimizer
            results_df: DataFrame with participant results
            participant_manager: Optional ParticipantDataManager instance to use
            
        Returns:
            Self for method chaining
        """
        # Handle invalid or empty input
        if optimizer_circles is None:
            self.logger.error("Cannot initialize with None optimizer_circles")
            return self
            
        # Set up participant manager - use provided instance, session state, or create new one
        import streamlit as st
        self.participant_manager = participant_manager
        
        # If not provided directly, try to get from session state
        if self.participant_manager is None and hasattr(st, 'session_state'):
            if 'participant_data_manager' in st.session_state:
                self.participant_manager = st.session_state.participant_data_manager
                self.logger.info("Using ParticipantDataManager from session state")
            else:
                self.logger.info("No ParticipantDataManager found in session state")
        
        # If still None and we have results_df, create a new instance
        if self.participant_manager is None and results_df is not None:
            from utils.participant_data_manager import ParticipantDataManager
            self.logger.info("Creating new ParticipantDataManager instance from results_df")
            self.participant_manager = ParticipantDataManager().initialize_from_dataframe(results_df)
            
            # Store in session state if available
            if hasattr(st, 'session_state'):
                st.session_state.participant_data_manager = self.participant_manager
                self.logger.info("Stored ParticipantDataManager in session state")
                
        if self.participant_manager is None:
            self.logger.warning("No ParticipantDataManager available for member data access")
        
        # Ensure we have a list of dictionaries to work with
        processed_circles = []
        try:
            # Process the optimizer_circles based on type
            if isinstance(optimizer_circles, pd.DataFrame):
                self.logger.info(f"Converting DataFrame with {len(optimizer_circles)} rows to list of dictionaries")
                # Convert DataFrame to list of dictionaries
                processed_circles = optimizer_circles.to_dict('records')
            elif isinstance(optimizer_circles, list):
                # Process each item in the list
                for circle in optimizer_circles:
                    if isinstance(circle, dict):
                        processed_circles.append(circle)
                    elif isinstance(circle, str):
                        # Handle string item - convert to dict with circle_id
                        self.logger.warning(f"Found string circle item: {circle}, converting to dict")
                        processed_circles.append({'circle_id': circle})
                    else:
                        self.logger.warning(f"Found invalid circle item type: {type(circle)}, skipping")
            elif isinstance(optimizer_circles, dict):
                # Single dictionary - add to list
                processed_circles = [optimizer_circles]
            else:
                # Try to handle other types as a last resort
                self.logger.warning(f"Unexpected optimizer_circles type: {type(optimizer_circles)}")
                # Try stringification as last resort
                processed_circles = [{'circle_id': str(optimizer_circles)}]
        except Exception as e:
            self.logger.error(f"Error processing optimizer_circles: {str(e)}")
            processed_circles = []
        
        self.logger.info(f"Processed {len(processed_circles)} circles from optimizer input")
        
        # Store reference to original optimizer circles for comparison and debugging
        self.optimizer_circles = processed_circles
        
        # Store reference to results DataFrame for member lookups
        self.results_df = results_df
        
        # Debug output to check target test circles in optimizer data
        test_circle_ids = ['IP-BOS-04', 'IP-BOS-05']
        print(f"\n🔍 CHECKING TEST CIRCLES IN OPTIMIZER DATA:")
        for test_id in test_circle_ids:
            test_circles = [c for c in processed_circles if c.get('circle_id') == test_id]
            if test_circles:
                tc = test_circles[0]
                print(f"  Found {test_id} in optimizer data: ")
                print(f"    max_additions: {tc.get('max_additions', 'N/A')}")
                print(f"    always_hosts: {tc.get('always_hosts', 'N/A')}")
                print(f"    sometimes_hosts: {tc.get('sometimes_hosts', 'N/A')}")
                print(f"    members: {len(tc.get('members', []))} members")
            else:
                print(f"  ⚠️ {test_id} not found in optimizer data")
        
        # Initialize circles dictionary from processed circles
        self.circles = {}
        for circle in processed_circles:
            try:
                # Safely get circle_id
                circle_id = None
                if isinstance(circle, dict) and 'circle_id' in circle:
                    circle_id = circle['circle_id']
                elif hasattr(circle, 'get'):
                    circle_id = circle.get('circle_id')
                elif hasattr(circle, 'circle_id'):
                    circle_id = circle.circle_id
                
                if circle_id:
                    # Ensure we store a dictionary
                    if isinstance(circle, dict):
                        self.circles[circle_id] = circle.copy()  # Create a deep copy to avoid reference issues
                    else:
                        # Convert to dict if needed
                        self.logger.warning(f"Converting non-dict circle to dictionary: {circle}")
                        if hasattr(circle, '__dict__'):
                            self.circles[circle_id] = circle.__dict__.copy()
                        else:
                            self.circles[circle_id] = {'circle_id': circle_id}
                else:
                    self.logger.warning(f"Found circle without circle_id: {circle}")
            except Exception as e:
                self.logger.error(f"Error processing circle: {str(e)} - {circle}")
                continue
        
        self._initialized = True
        self.logger.info(f"Successfully initialized {len(self.circles)} circles")
        
        # Perform initial normalization and validation
        self.normalize_metadata()
        self.validate_circles()
        
        return self
    
    def normalize_metadata(self) -> None:
        """Normalize all circle metadata to ensure consistency"""
        self.logger.info("Normalizing circle metadata")
        
        # Apply normalizations
        self.normalize_host_values()
        self.normalize_numeric_fields()
        self.fill_missing_metadata()
        
        self.logger.info("Circle metadata normalization complete")
    
    def normalize_host_values(self) -> None:
        """Normalize host values to ensure consistent counting"""
        self.logger.info("Normalizing host values")
        
        # Initialize counters for statistics
        always_fixed = 0
        sometimes_fixed = 0
        target_circles = ['IP-BOS-04', 'IP-BOS-05']  # Specific circles we're troubleshooting
        
        print("\n🔍 HOST NORMALIZATION: Checking host counts across all circles")
        
        for circle_id, circle in self.circles.items():
            # Count hosts using results_df as source of truth
            if self.results_df is not None and 'members' in circle and circle['members']:
                # Convert to list if needed
                members_list = self._ensure_list(circle['members'])
                
                # Extra debug for target circles
                is_target = circle_id in target_circles
                if is_target:
                    print(f"\n🔍 DETAILED HOST CHECK FOR {circle_id}:")
                    print(f"  Found {len(members_list)} members in this circle")
                    print(f"  Member IDs: {members_list}")
                
                # Count by analyzing each member
                # CRITICAL FIX: Pass circle_id to the host counting function for better debugging
                always_hosts, sometimes_hosts = self._count_hosts_from_members(members_list, circle_id)
                
                # Update counts if they differ from current values
                always_before = circle.get('always_hosts', 0)
                sometimes_before = circle.get('sometimes_hosts', 0)
                
                # Additional debug for target circles
                if is_target:
                    print(f"  Current host counts: {always_before} Always, {sometimes_before} Sometimes")
                    print(f"  Calculated host counts: {always_hosts} Always, {sometimes_hosts} Sometimes")
                
                # Check for significant differences that might indicate a problem
                if always_before > 0 and always_hosts == 0:
                    print(f"  ⚠️ WARNING: Circle {circle_id} had {always_before} Always Hosts before but 0 now")
                    # Add more detailed debugging for this case
                    if self.results_df is not None and len(members_list) > 0:
                        print(f"  🔍 MEMBER HOST STATUS CHECK:")
                        for member_id in members_list:
                            member_rows = self.results_df[self.results_df['Encoded ID'] == member_id]
                            if not member_rows.empty:
                                host_col = None
                                for col in ['host', 'Host', 'willing_to_host']:
                                    if col in self.results_df.columns:
                                        host_col = col
                                        break
                                if host_col:
                                    host_status = member_rows.iloc[0][host_col]
                                    print(f"    Member {member_id}: host_status='{host_status}'")
                                else:
                                    print(f"    Member {member_id}: No host column found")
                            else:
                                print(f"    Member {member_id}: Not found in results DataFrame")
                
                if always_before != always_hosts:
                    circle['always_hosts'] = always_hosts
                    always_fixed += 1
                    if is_target or (always_before == 0 and always_hosts > 0) or (always_before > 0 and always_hosts == 0):
                        print(f"  ✅ FIXED: Updated always_hosts for {circle_id}: {always_before} → {always_hosts}")
                    self.logger.debug(f"Fixed always_hosts for {circle_id}: {always_before} → {always_hosts}")
                
                if sometimes_before != sometimes_hosts:
                    circle['sometimes_hosts'] = sometimes_hosts
                    sometimes_fixed += 1
                    if is_target:
                        print(f"  ✅ FIXED: Updated sometimes_hosts for {circle_id}: {sometimes_before} → {sometimes_hosts}")
                    self.logger.debug(f"Fixed sometimes_hosts for {circle_id}: {sometimes_before} → {sometimes_hosts}")
        
        print(f"Host normalization complete: Fixed {always_fixed} always_hosts and {sometimes_fixed} sometimes_hosts values")
        self.logger.info(f"Host normalization complete: Fixed {always_fixed} always_hosts and {sometimes_fixed} sometimes_hosts values")
    
    def _count_hosts_from_members(self, member_ids: List[str], circle_id: str = None) -> tuple:
        """Count always and sometimes hosts from member list with standardized detection"""
        from utils.data_standardization import normalize_host_status
        from utils.feature_flags import get_flag
        
        always_hosts = 0
        sometimes_hosts = 0
        
        # Add special circle debugging for our test circles
        test_circle_ids = ['IP-BOS-04', 'IP-BOS-05', 'IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']
        is_test_circle = circle_id in test_circle_ids if circle_id else False
        
        # ENHANCED DEBUG: More descriptive debug information
        debug_prefix = f"[CIRCLE: {circle_id}]" if circle_id else "[UNKNOWN CIRCLE]"
        
        # SPECIAL DEBUG: Targeted diagnostics for test circles
        if is_test_circle:
            print(f"\n🔍🔍🔍 SPECIAL TEST CIRCLE HOST DEBUG FOR {circle_id} 🔍🔍🔍")
            print(f"  {debug_prefix} Found {len(member_ids)} members in member_ids list")
            if len(member_ids) <= 15:  # Only print full list if it's reasonably short
                print(f"  {debug_prefix} Member IDs: {member_ids}")
        
        # First try using ParticipantDataManager if available
        if hasattr(self, 'participant_manager') and self.participant_manager is not None:
            if is_test_circle:
                print(f"  {debug_prefix} Using ParticipantDataManager for host status detection")
                
            # Process each member using the manager
            host_statuses = []
            all_host_statuses = []
            
            for member_id in member_ids:
                try:
                    # Get host status directly from manager (which handles standardization)
                    host_status = self.participant_manager.get_participant_host_status(member_id)
                    all_host_statuses.append(f"{member_id}: {host_status}")
                    
                    # Count by category
                    if host_status == "always":
                        always_hosts += 1
                        host_statuses.append(f"{member_id}: ALWAYS")
                    elif host_status == "sometimes":
                        sometimes_hosts += 1
                        host_statuses.append(f"{member_id}: SOMETIMES")
                    else:
                        host_statuses.append(f"{member_id}: NEVER")
                        
                except Exception as e:
                    if is_test_circle:
                        print(f"  {debug_prefix} ⚠️ Error getting host status for member {member_id}: {str(e)}")
            
            if is_test_circle:
                print(f"  {debug_prefix} 📊 Host distribution via ParticipantDataManager:")
                print(f"    - Always Hosts: {always_hosts}")
                print(f"    - Sometimes Hosts: {sometimes_hosts}")
                if len(host_statuses) <= 15:
                    print(f"    - Member host statuses: {', '.join(host_statuses)}")
                
            # Only fall back to results_df if manager failed to find hosts
            if always_hosts > 0 or sometimes_hosts > 0:
                return always_hosts, sometimes_hosts
            else:
                if is_test_circle:
                    print(f"  {debug_prefix} ⚠️ ParticipantDataManager found no hosts, falling back to results_df")
        elif is_test_circle:
            print(f"  {debug_prefix} No ParticipantDataManager available, using results_df")
        
        # Fall back to traditional approach with results_df
        if self.results_df is None:
            if is_test_circle:
                print(f"  {debug_prefix} ⚠️ ERROR: results_df is None, cannot count hosts")
            return always_hosts, sometimes_hosts
        
        # Ensure Encoded ID column exists
        if 'Encoded ID' not in self.results_df.columns:
            self.logger.warning(f"{debug_prefix} Cannot count hosts: 'Encoded ID' column missing from results DataFrame")
            if is_test_circle:
                print(f"  {debug_prefix} ⚠️ ERROR: 'Encoded ID' column missing from results DataFrame")
                print(f"  {debug_prefix} Available columns: {self.results_df.columns.tolist()}")
            return always_hosts, sometimes_hosts
        
        # First check if host_status_standardized exists and should be used
        use_standardized = get_flag('use_standardized_host_status')
        standardized_col_exists = 'host_status_standardized' in self.results_df.columns
        
        if use_standardized and standardized_col_exists:
            host_col = 'host_status_standardized'
            if is_test_circle:
                print(f"  {debug_prefix} ✅ Using standardized host status column")
        else:
            # Host column may have different names, try to find it
            host_col = None
            for col in ['host', 'Host', 'willing_to_host']: 
                if col in self.results_df.columns:
                    host_col = col
                    break
        
        if not host_col:
            self.logger.warning(f"{debug_prefix} Cannot count hosts: No host column found in results DataFrame")
            if is_test_circle:
                print(f"  {debug_prefix} ⚠️ ERROR: No host column found in results DataFrame")
                print(f"  {debug_prefix} Available columns: {self.results_df.columns.tolist()}")
            return always_hosts, sometimes_hosts
        
        # COUNT: Count hosts from results DataFrame with enhanced robustness
        missing_members = 0
        found_members = 0
        
        # Debug check - more comprehensive for test circles
        if is_test_circle:
            print(f"\n{debug_prefix} 🔍 DETAILED HOST STATUS DETECTION:")
            print(f"  {debug_prefix} Looking up {len(member_ids)} members in results_df with {len(self.results_df)} rows")
            print(f"  {debug_prefix} Using '{host_col}' column for host status")
        
        # Track all host status values for debugging
        all_host_statuses = {}
        always_host_values = []
        sometimes_host_values = []
        
        # If we're using standardized host status, the counting is much simpler
        if host_col == 'host_status_standardized':
            if is_test_circle:
                print(f"  {debug_prefix} Using standardized host status values (ALWAYS/SOMETIMES/NEVER)")
                
            # Get the member data
            member_data = self.results_df[self.results_df['Encoded ID'].isin(member_ids)]
            # Count based on standardized values
            always_hosts = (member_data['host_status_standardized'] == 'ALWAYS').sum()
            sometimes_hosts = (member_data['host_status_standardized'] == 'SOMETIMES').sum()
            
            found_members = len(member_data)
            missing_members = len(member_ids) - found_members
            
            # Debug output
            if is_test_circle:
                print(f"  {debug_prefix} Found {found_members} of {len(member_ids)} members in results DataFrame")
                if missing_members > 0:
                    print(f"  {debug_prefix} ⚠️ Could not find {missing_members} members in results DataFrame")
                print(f"  {debug_prefix} Counted {always_hosts} ALWAYS hosts and {sometimes_hosts} SOMETIMES hosts")
        else:
            # Legacy counting with on-the-fly normalization if feature flag is enabled
            # CRITICAL FIX: Test the optimization detection as well for IP-BOS-04 and IP-BOS-05
            # This ensures both "Always Host" and "Always" are correctly detected
            if is_test_circle:
                print(f"\n{debug_prefix} 🧪 TESTING HOST DETECTION STRINGS:")
                test_values = [
                    "Always", "Always Host", "always", "ALWAYS", "always host", 
                    "Sometimes", "Sometimes Host", "sometimes", "SOMETIMES", "sometimes host"
                ]
                
                for test_val in test_values:
                    # If normalize_on_the_fly is enabled, use the standardization function
                    normalize_on_the_fly = get_flag('use_standardized_host_status')
                    
                    if normalize_on_the_fly:
                        normalized = normalize_host_status(test_val)
                        print(f"  {debug_prefix} Testing value '{test_val}': Normalized to '{normalized}'")
                    else:
                        host_lower = test_val.lower()
                        detection = "UNDETECTED"
                        
                        # Test with our traditional pattern matching
                        if ('always' in host_lower) or host_lower == 'always':
                            detection = "✅ Would be counted as ALWAYS HOST"
                        elif ('sometimes' in host_lower) or host_lower == 'sometimes':
                            detection = "✅ Would be counted as SOMETIMES HOST"
                        
                        print(f"  {debug_prefix} Testing value '{test_val}': {detection}")

        
        for member_id in member_ids:
            # Look up this member in results_df
            try:
                member_rows = self.results_df[self.results_df['Encoded ID'] == member_id]
                
                if not member_rows.empty:
                    found_members += 1
                    # Get host status
                    host_status = member_rows.iloc[0][host_col]
                    all_host_statuses[member_id] = str(host_status) if not pd.isna(host_status) else "None/NaN"
                    
                    # CRITICAL FIX: Enhanced host detection with comprehensive pattern matching
                    
                    # First handle None/NaN values
                    if pd.isna(host_status) or host_status is None:
                        if is_test_circle:
                            print(f"  {debug_prefix} Member {member_id}: host_status is None/NaN - Not counted as host")
                        continue
                    
                    # Convert to string for consistent comparison
                    if not isinstance(host_status, str):
                        # Handle boolean and numeric values
                        if host_status in [True, 1]:
                            always_hosts += 1
                            always_host_values.append(str(host_status))
                            if is_test_circle:
                                print(f"  {debug_prefix} Member {member_id}: host_status={host_status} (type: {type(host_status).__name__})")
                                print(f"    {debug_prefix} ✅ Counted as ALWAYS HOST (boolean/numeric match)")
                            continue
                        else:
                            # Convert non-string non-boolean to string for further processing
                            host_status = str(host_status)
                    
                    # Now we're sure it's a string, normalize for comparison
                    host_lower = host_status.lower().strip()
                    
                    # Enhanced debug for test circles
                    if is_test_circle:
                        print(f"  {debug_prefix} Member {member_id}: host_status='{host_status}' (type: {type(host_status).__name__})")
                    
                    # IMPROVED MATCHING: More comprehensive pattern matching for both Always and Sometimes hosts
                    # Correctly interpret "Always Host" as well as "Always"
                    if ('always' in host_lower) or host_lower == 'always' or host_lower in ['yes', 'true', 'true.0']:
                        always_hosts += 1
                        always_host_values.append(host_status)
                        if is_test_circle:
                            print(f"    {debug_prefix} ✅ Counted as ALWAYS HOST")
                    elif ('sometimes' in host_lower) or host_lower == 'sometimes' or host_lower in ['maybe']:
                        sometimes_hosts += 1
                        sometimes_host_values.append(host_status)
                        if is_test_circle:
                            print(f"    {debug_prefix} ✅ Counted as SOMETIMES HOST")
                    else:
                        if is_test_circle:
                            print(f"    {debug_prefix} ℹ️ Not counted as host (unrecognized value: '{host_status}')")
                else:
                    missing_members += 1
                    if is_test_circle:
                        print(f"  {debug_prefix} ⚠️ Member {member_id} not found in results_df")
            except Exception as e:
                print(f"  {debug_prefix} ⚠️ Error processing member {member_id}: {str(e)}")
                missing_members += 1
        
        if missing_members > 0:
            self.logger.warning(f"{debug_prefix} Could not find {missing_members} members in results DataFrame")
            if is_test_circle:
                print(f"  {debug_prefix} ⚠️ Could not find {missing_members} out of {len(member_ids)} members")
        
        # No special case overrides - rely on data-driven approach only
        
        # Final host counts summary with enhanced debugging for test circles
        if is_test_circle:
            print(f"\n{debug_prefix} 🔍 FINAL HOST COUNT SUMMARY:")
            print(f"  {debug_prefix} Members found: {found_members} out of {len(member_ids)}")
            print(f"  {debug_prefix} All host values found: {all_host_statuses}")
            print(f"  {debug_prefix} 'Always' host values: {always_host_values}")
            print(f"  {debug_prefix} 'Sometimes' host values: {sometimes_host_values}")
            print(f"  {debug_prefix} FINAL COUNTS: {always_hosts} Always Hosts, {sometimes_hosts} Sometimes Hosts")
        else:
            print(f"  Final counts: {always_hosts} Always Hosts, {sometimes_hosts} Sometimes Hosts")
        
        return always_hosts, sometimes_hosts
    
    def normalize_numeric_fields(self) -> None:
        """Ensure numeric fields are actual numbers"""
        numeric_fields = ['member_count', 'new_members', 'continuing_members', 
                         'always_hosts', 'sometimes_hosts', 'max_additions']
        
        for circle_id, circle in self.circles.items():
            for field in numeric_fields:
                if field in circle:
                    # Convert to int if possible
                    try:
                        current_value = circle[field]
                        if pd.isna(current_value):
                            circle[field] = 0
                        else:
                            circle[field] = int(current_value)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Could not convert {field}='{circle[field]}' to int for {circle_id}, setting to 0")
                        circle[field] = 0
    
    def fill_missing_metadata(self) -> None:
        """Fill in missing metadata fields with default or derived values"""
        for circle_id, circle in self.circles.items():
            # Always ensure basic fields exist
            if 'circle_id' not in circle:
                circle['circle_id'] = circle_id
            
            # Derive region and subregion from circle_id if missing
            if 'region' not in circle or not circle['region']:
                circle['region'] = self._extract_region_from_id(circle_id)
            
            if 'subregion' not in circle or not circle['subregion']:
                circle['subregion'] = self._extract_subregion_from_id(circle_id)
            
            # Mark as active by default if not already set
            if 'active' not in circle:
                # If this circle has been replaced by splits, it should be inactive
                if circle.get('replaced_by_splits', False):
                    circle['active'] = False
                else:
                    circle['active'] = True
            
            # Check if this is a split circle and update the tracking dictionaries
            if 'SPLIT' in circle_id:
                # Identify and track the relationship between original and split circles
                if 'original_circle_id' in circle and circle['original_circle_id']:
                    original_id = circle['original_circle_id']
                    self.split_circles[circle_id] = original_id
                    
                    # Add to original circles tracking if it doesn't exist
                    if original_id not in self.original_circles:
                        self.original_circles[original_id] = []
                    
                    # Add this split circle to the original's list if not already there
                    if circle_id not in self.original_circles[original_id]:
                        self.original_circles[original_id].append(circle_id)
                        
                # Ensure the split circle is marked as such
                circle['is_split_circle'] = True
                
                # Set split_letter if not already set (e.g., A, B, C)
                if 'split_letter' not in circle and circle_id[-1].isalpha():
                    circle['split_letter'] = circle_id[-1]
                
                # Set max_additions if not already set (split circles can grow to 8 members max)
                if 'max_additions' not in circle:
                    # If we have member count, calculate based on that
                    if 'member_count' in circle and isinstance(circle['member_count'], int):
                        circle['max_additions'] = max(0, 8 - circle['member_count'])
                    else:
                        # Default to allowing 3 new members
                        circle['max_additions'] = 3
            
            # Ensure member_count matches actual members count if available
            if 'members' in circle and isinstance(circle['members'], list):
                actual_count = len(circle['members'])
                if 'member_count' not in circle or circle['member_count'] != actual_count:
                    self.logger.debug(f"Updating member_count for {circle_id} from {circle.get('member_count', 'None')} to {actual_count}")
                    circle['member_count'] = actual_count
    
    def _extract_region_from_id(self, circle_id: str) -> str:
        """Extract region from circle ID"""
        if not circle_id or not isinstance(circle_id, str):
            return "Unknown"
        
        # Parse IP-XXX-## format
        parts = circle_id.split('-')
        if len(parts) >= 2:
            region_code = parts[1]
            # Map common region codes to full names
            region_map = {
                'SFO': 'San Francisco',
                'NYC': 'New York',
                'BOS': 'Boston',
                'CHI': 'Chicago',
                'SEA': 'Seattle',
                'LA': 'Los Angeles',
                'HOU': 'Houston',
                'ATL': 'Atlanta',
                'DEN': 'Denver',
                'DC': 'Washington DC',
                'MAR': 'Marin County',
                'PEN': 'Peninsula',
                'EBA': 'East Bay',
                'SIN': 'Singapore'
                # Add more mappings as needed
            }
            return region_map.get(region_code, region_code)
        
        return "Unknown"
    
    def _extract_subregion_from_id(self, circle_id: str) -> str:
        """Extract subregion from circle ID if available"""
        if not circle_id or not isinstance(circle_id, str):
            return "Unknown"
        
        # For some regions, the subregion may be encoded in the circle number
        # This is application-specific and would need to be customized
        # For now, return the region as the default subregion
        return self._extract_region_from_id(circle_id)
    
    def validate_circles(self) -> None:
        """Validate all circles for data consistency"""
        self.logger.info("Validating circle data consistency")
        
        # Validate max_additions against new members
        self.validate_max_additions()
        
        # Validate other aspects as needed
        # ...
        
        self.logger.info("Circle validation complete")
    
    def add_or_update_circle(self, circle_id: str, circle_data: dict) -> None:
        """
        Add a new circle or update an existing one.
        
        Args:
            circle_id: The ID of the circle to add or update
            circle_data: Dictionary containing circle metadata
        """
        if not circle_id:
            self.logger.error("Cannot add circle with empty ID")
            return
            
        # If this is a new circle, add it
        if circle_id not in self.circles:
            self.circles[circle_id] = circle_data.copy()
            self.logger.info(f"Added new circle {circle_id}")
        else:
            # Otherwise update the existing circle
            # Start with a copy of existing data
            existing_data = self.circles[circle_id].copy()
            
            # Apply updates from new data
            for key, value in circle_data.items():
                existing_data[key] = value
                
            # Store updated data
            self.circles[circle_id] = existing_data
            self.logger.info(f"Updated circle {circle_id}")
            
        # Check if this is a split circle and update tracking
        if 'is_split_circle' in circle_data and circle_data['is_split_circle']:
            print(f"🔄 Adding/updating split circle {circle_id}")
            
            # Track the split circle
            if 'original_circle_id' in circle_data and circle_data['original_circle_id']:
                original_id = circle_data['original_circle_id']
                self.split_circles[circle_id] = original_id
                
                # Add to original circles tracking if it doesn't exist
                if original_id not in self.original_circles:
                    self.original_circles[original_id] = []
                
                # Add this split circle to the original's list if not already there
                if circle_id not in self.original_circles[original_id]:
                    self.original_circles[original_id].append(circle_id)
                    
                print(f"✅ Updated split circle tracking for {circle_id} (original: {original_id})")
            else:
                print(f"⚠️ Split circle {circle_id} does not have original_circle_id")
                
        # Normalize the data to ensure consistency
        self.normalize_metadata()
    
    def remove_circle(self, circle_id: str) -> bool:
        """
        Remove a circle from the manager.
        
        Args:
            circle_id: The ID of the circle to remove
            
        Returns:
            bool: True if the circle was removed, False if it was not found
        """
        if circle_id not in self.circles:
            self.logger.warning(f"Cannot remove non-existent circle {circle_id}")
            return False
            
        # Check if this is a split circle
        if circle_id in self.split_circles:
            original_id = self.split_circles[circle_id]
            
            # Remove from the original's list
            if original_id in self.original_circles and circle_id in self.original_circles[original_id]:
                self.original_circles[original_id].remove(circle_id)
                
                # If this was the last split circle, remove the original from tracking
                if not self.original_circles[original_id]:
                    del self.original_circles[original_id]
                    
            # Remove from split circles tracking
            del self.split_circles[circle_id]
            
        # Check if this is an original circle with splits
        if circle_id in self.original_circles:
            # Usually we want to remove all split circles associated with this original
            split_ids = self.original_circles[circle_id].copy()
            
            for split_id in split_ids:
                if split_id in self.circles:
                    del self.circles[split_id]
                if split_id in self.split_circles:
                    del self.split_circles[split_id]
                    
            # Remove from original circles tracking
            del self.original_circles[circle_id]
            
        # Remove the circle itself
        del self.circles[circle_id]
        self.logger.info(f"Removed circle {circle_id}")
        return True
    
    def has_circle(self, circle_id: str) -> bool:
        """
        Check if a circle exists in the manager.
        
        Args:
            circle_id: The ID of the circle to check
            
        Returns:
            bool: True if the circle exists, False otherwise
        """
        return circle_id in self.circles
    
    def is_split_circle(self, circle_id: str) -> bool:
        """
        Check if a circle is a split circle.
        
        Args:
            circle_id: The ID of the circle to check
            
        Returns:
            bool: True if the circle is a split circle, False otherwise
        """
        return circle_id in self.split_circles
        
    def get_original_circle_id(self, split_circle_id: str) -> str:
        """
        Get the original circle ID for a split circle.
        
        Args:
            split_circle_id: The ID of the split circle
            
        Returns:
            str: The original circle ID, or None if not a split circle
        """
        return self.split_circles.get(split_circle_id, None)
        
    def get_split_circle_ids(self, original_circle_id: str) -> list:
        """
        Get the split circle IDs for an original circle.
        
        Args:
            original_circle_id: The ID of the original circle
            
        Returns:
            list: List of split circle IDs, or empty list if not an original circle
        """
        return self.original_circles.get(original_circle_id, [])
    
    def get_circles_dataframe(self, include_inactive=False):
        """
        Get all circles as a pandas DataFrame.
        
        Args:
            include_inactive: Whether to include circles that have been replaced by splits
            
        Returns:
            DataFrame containing all circle data
        """
        import pandas as pd
        
        # Get all circles with enhanced metadata
        circle_data = self.get_all_circles_enhanced(include_inactive)
        
        # Create DataFrame
        if circle_data:
            return pd.DataFrame(circle_data)
        
        # Return empty DataFrame if no data
        return pd.DataFrame()
    
    def get_all_circles_enhanced(self, include_inactive=False):
        """
        Get all circle data with enhanced metadata for display.
        
        Args:
            include_inactive: Whether to include circles that have been replaced by splits
            
        Returns:
            List of dictionaries with enhanced circle data
        """
        enhanced_circles = []
        
        for circle_id, circle in self.circles.items():
            # Skip inactive circles (original circles that have been split) if not including them
            # FIXED: Check for is_active=False OR replaced_by_splits=True to determine inactive status
            is_inactive = (
                circle.get('is_active') is False or  # explicitly inactive
                circle.get('active') is False or     # legacy inactive flag
                circle.get('replaced_by_splits', False)  # original circles that were split
            )
            
            if not include_inactive and is_inactive:
                continue
                
            # Create a copy to avoid modifying the original
            enhanced_circle = circle.copy()
            
            # Calculate additional metrics
            members_list = self._ensure_list(circle.get('members', []))
            enhanced_circle['member_count'] = len(members_list)
            
            # Normalize host counts
            if 'always_hosts' not in enhanced_circle or enhanced_circle['always_hosts'] is None:
                enhanced_circle['always_hosts'] = 0
            if 'sometimes_hosts' not in enhanced_circle or enhanced_circle['sometimes_hosts'] is None:
                enhanced_circle['sometimes_hosts'] = 0
                
            # Add split status information
            enhanced_circle['is_split_circle'] = circle_id in self.split_circles
            if enhanced_circle['is_split_circle']:
                enhanced_circle['original_circle_id'] = self.split_circles[circle_id]
                # Add split letter (A, B, C, etc.) extracted from ID
                if "-SPLIT-" in circle_id:
                    # Capture the split letter (last character)
                    enhanced_circle['split_letter'] = circle_id[-1]
                    # Create a human-readable split status for display
                    enhanced_circle['split_status'] = f"Split {enhanced_circle['split_letter']}"
                else:
                    enhanced_circle['split_status'] = "Split"
                
                # FIXED: Ensure max_additions is set for split circles
                # Split circles can add members up to a maximum of 8 total
                if 'max_additions' not in enhanced_circle or enhanced_circle['max_additions'] is None:
                    max_total = 8  # Maximum size for a split circle
                    current_count = enhanced_circle['member_count']
                    enhanced_circle['max_additions'] = max(0, max_total - current_count)
                    print(f"  ✅ Set max_additions={enhanced_circle['max_additions']} for split circle {circle_id}")
            
            enhanced_circle['has_splits'] = circle_id in self.original_circles and len(self.original_circles[circle_id]) > 0
            if enhanced_circle['has_splits']:
                enhanced_circle['split_circle_ids'] = self.original_circles[circle_id]
                # Mark original circles as inactive if they have splits
                enhanced_circle['split_status'] = "Original (Split)"
                
            # Add circle to results
            enhanced_circles.append(enhanced_circle)
            
        return enhanced_circles
        
    def synchronize_metadata(self, circles_df=None, results_df=None, split_summary=None):
        """
        Synchronize metadata across all circle data structures to ensure consistency.
        This method serves as the central synchronization point for circle metadata.
        
        Args:
            circles_df: DataFrame containing circle data (matched_circles)
            results_df: DataFrame containing participant results
            split_summary: Dictionary containing split circle summary data
            
        Returns:
            tuple: (updated_circles_df, has_changes) - DataFrame with synchronized metadata and a boolean indicating if changes were made
        """
        self.logger.info("Starting metadata synchronization process")
        has_changes = False
        
        # Step 1: Update metadata manager with any new circle data
        if circles_df is not None:
            self.logger.info(f"Synchronizing metadata from circles DataFrame with {len(circles_df)} circles")
            # We don't want to reinitialize completely, just update existing circles and add new ones
            self._update_existing_circles(circles_df)
            has_changes = True
            
        # Step 2: Update metadata manager with any new results data
        if results_df is not None:
            self.logger.info(f"Synchronizing metadata from results DataFrame with {len(results_df)} participants")
            self.results_df = results_df
            has_changes = True
            
        # Step 3: Update metadata manager with split circle information
        if split_summary is not None and 'split_details' in split_summary:
            self.logger.info(f"Synchronizing metadata from split summary with {len(split_summary['split_details'])} splits")
            self._process_split_summary(split_summary)
            has_changes = True
        
        # Step 4: Validate metadata consistency and fix discrepancies
        validation_results = self._validate_metadata_consistency()
        if validation_results['discrepancies_found']:
            self.logger.warning(f"Found {validation_results['total_discrepancies']} metadata discrepancies")
            self._fix_metadata_discrepancies(validation_results)
            has_changes = True
        
        # Step 5: Return updated circle data
        if has_changes:
            # Get updated circles DataFrame
            updated_circles_df = self.get_circles_dataframe()
            self.logger.info(f"Metadata synchronization complete - returning updated circles DataFrame with {len(updated_circles_df)} circles")
            return updated_circles_df, True
        else:
            self.logger.info("No changes made during metadata synchronization")
            return circles_df, False
    
    def _update_existing_circles(self, circles_df):
        """
        Update existing circles with data from circles DataFrame without full reinitialization.
        
        Args:
            circles_df: DataFrame containing circle data
        """
        if circles_df is None or len(circles_df) == 0:
            self.logger.warning("Empty circles DataFrame provided for update")
            return
            
        # Convert to list of dictionaries if it's a DataFrame
        circle_dicts = circles_df.to_dict('records') if hasattr(circles_df, 'to_dict') else circles_df
            
        # Update existing circles and add new ones
        for circle_data in circle_dicts:
            circle_id = circle_data.get('circle_id')
            if not circle_id:
                self.logger.warning(f"Skipping circle data without circle_id: {circle_data}")
                continue
                
            if circle_id in self.circles:
                # Update existing circle
                self.logger.debug(f"Updating existing circle: {circle_id}")
                for key, value in circle_data.items():
                    # Don't overwrite critical fields with None or empty values
                    if value is not None and value != '' and key != 'circle_id':
                        self.circles[circle_id][key] = value
            else:
                # Add new circle
                self.logger.debug(f"Adding new circle: {circle_id}")
                self.circles[circle_id] = circle_data.copy()
    
    def _process_split_summary(self, split_summary):
        """
        Process split circle summary data to update split circle tracking.
        
        Args:
            split_summary: Dictionary containing split circle summary data
        """
        if not split_summary or 'split_details' not in split_summary:
            self.logger.warning("Invalid split summary provided")
            return
            
        # Process each split
        for detail in split_summary['split_details']:
            original_id = detail.get('original_circle_id')
            new_circle_ids = detail.get('new_circle_ids', [])
            
            if not original_id or not new_circle_ids:
                self.logger.warning(f"Invalid split detail: {detail}")
                continue
                
            # Update original_circles tracking
            self.original_circles[original_id] = new_circle_ids
            
            # Update split_circles tracking
            for new_id in new_circle_ids:
                self.split_circles[new_id] = original_id
                
            # Mark original circle as inactive if it exists, but ensure split circles remain active
            if original_id in self.circles:
                self.logger.info(f"Marking original circle {original_id} as inactive (replaced by splits)")
                self.circles[original_id]['is_active'] = False  # Original is inactive
                self.circles[original_id]['replaced_by_splits'] = True
                self.circles[original_id]['split_into'] = new_circle_ids  # Store which circles it was split into
                
            # Ensure all split circles have proper metadata
            for i, new_id in enumerate(new_circle_ids):
                # Add null check to ensure detail and detail['members'] exist and are valid
                if new_id not in self.circles and detail is not None and 'members' in detail and i < len(detail.get('members', [])):
                    # Get members with null safety
                    members_list = detail.get('members', [])
                    if i < len(members_list) and members_list[i] is not None:
                        member_list = members_list[i]
                        # Calculate member count safely
                        member_count = len(member_list) if isinstance(member_list, list) else 0
                        # Create new circle entry if it doesn't exist with explicit eligibility flags
                        self.circles[new_id] = {
                            'circle_id': new_id,
                            'is_split_circle': True,
                            'original_circle_id': original_id,
                            'is_active': True,  # Split circles are active
                            'eligible_for_new_members': True,  # Split circles are eligible for new members
                            'members': member_list,
                            'member_count': member_count,
                            'max_additions': max(0, 8 - member_count),  # Allow growth up to 8 total
                            'split_letter': new_id[-1] if new_id[-1].isalpha() else "",  # Store split letter (A, B, C)
                            'split_index': i  # Store split index (0, 1, 2)
                        }
                    else:
                        # Create entry with empty members if data is incomplete
                        self.logger.warning(f"Incomplete member data for split circle {new_id} at index {i}")
                        self.circles[new_id] = {
                            'circle_id': new_id,
                            'is_split_circle': True,
                            'original_circle_id': original_id,
                            'is_active': True,  # Split circles are active
                            'eligible_for_new_members': True,  # Split circles are eligible
                            'members': [],
                            'member_count': 0,
                            'max_additions': 8,  # Empty circle can take 8 members
                            'split_letter': new_id[-1] if new_id[-1].isalpha() else "",  # Store split letter (A, B, C)
                            'split_index': i  # Store split index (0, 1, 2)
                        }
                    
                    # Copy metadata from original circle
                    if original_id in self.circles:
                        for key in ['region', 'subregion', 'meeting_time']:
                            if key in self.circles[original_id]:
                                self.circles[new_id][key] = self.circles[original_id][key]
                                
                    # Add host information if available in the split summary
                    # Add null checks for 'always_hosts' access
                    if 'always_hosts' in detail and detail.get('always_hosts') is not None and i < len(detail.get('always_hosts', [])):
                        self.circles[new_id]['always_hosts'] = detail['always_hosts'][i]
                    else:
                        # Default to 0 if no host data available
                        self.circles[new_id]['always_hosts'] = 0
                        
                    # Add null checks for 'sometimes_hosts' access
                    if 'sometimes_hosts' in detail and detail.get('sometimes_hosts') is not None and i < len(detail.get('sometimes_hosts', [])):
                        self.circles[new_id]['sometimes_hosts'] = detail['sometimes_hosts'][i]
                    else:
                        # Default to 0 if no host data available
                        self.circles[new_id]['sometimes_hosts'] = 0
                        
                    # Add logging for debugging
                    self.logger.info(f"Added split circle {new_id} with {self.circles[new_id].get('member_count', 0)} members, " +
                                      f"{self.circles[new_id].get('always_hosts', 0)} always hosts, " +
                                      f"{self.circles[new_id].get('sometimes_hosts', 0)} sometimes hosts")
    
    def _validate_metadata_consistency(self):
        """
        Validate metadata consistency across all circle data.
        
        Returns:
            dict: Validation results including discrepancies found and total count
        """
        validation_results = {
            'discrepancies_found': False,
            'total_discrepancies': 0,
            'field_discrepancies': {},
            'circle_discrepancies': {}
        }
        
        # Check critical fields for each circle
        critical_fields = ['member_count', 'max_additions', 'always_hosts', 'sometimes_hosts']
        
        for circle_id, circle_data in self.circles.items():
            circle_discrepancies = []
            
            # Check member_count against length of members list
            if 'members' in circle_data and 'member_count' in circle_data:
                members = circle_data['members']
                if isinstance(members, list) and len(members) != circle_data['member_count']:
                    circle_discrepancies.append(f"member_count ({circle_data['member_count']}) != len(members) ({len(members)})")
                    
            # Check max_additions calculation
            if 'member_count' in circle_data and circle_data.get('is_active', True):
                # For active circles, max_additions should be 8 - member_count or 0 if member_count >= 8
                expected_max = max(0, 8 - circle_data['member_count']) 
                if 'max_additions' in circle_data and circle_data['max_additions'] != expected_max:
                    circle_discrepancies.append(f"max_additions ({circle_data['max_additions']}) != expected ({expected_max})")
            
            # If discrepancies found for this circle, add to results
            if circle_discrepancies:
                validation_results['discrepancies_found'] = True
                validation_results['total_discrepancies'] += len(circle_discrepancies)
                validation_results['circle_discrepancies'][circle_id] = circle_discrepancies
        
        return validation_results
    
    def _fix_metadata_discrepancies(self, validation_results):
        """
        Fix metadata discrepancies identified during validation.
        
        Args:
            validation_results: Dictionary containing validation results
        """
        if not validation_results['discrepancies_found']:
            return
            
        # Process each circle with discrepancies
        for circle_id, discrepancies in validation_results['circle_discrepancies'].items():
            if circle_id not in self.circles:
                continue
                
            circle_data = self.circles[circle_id]
            
            # Fix member_count based on members list
            if 'members' in circle_data and isinstance(circle_data['members'], list):
                circle_data['member_count'] = len(circle_data['members'])
                
            # Fix max_additions based on member_count
            if 'member_count' in circle_data and circle_data.get('is_active', True):
                # All circles (including split circles) have a max of 8 members total
                circle_data['max_additions'] = max(0, 8 - circle_data['member_count'])
                
                # For split circles, explicitly set eligibility flag
                if circle_data.get('is_split_circle', False):
                    circle_data['eligible_for_new_members'] = circle_data['member_count'] < 8
                    self.logger.info(f"Split circle {circle_id} eligibility set to {circle_data['eligible_for_new_members']} (member_count={circle_data['member_count']})")
                
            # Fix host counts - critical for circle eligibility
            if self.results_df is not None and 'members' in circle_data and circle_data['members']:
                # Get host counts from results DataFrame
                members_list = self._ensure_list(circle_data['members'])
                always_hosts, sometimes_hosts = self._count_hosts_from_members(members_list, circle_id)
                
                # Update circle metadata
                if 'always_hosts' not in circle_data or circle_data['always_hosts'] != always_hosts:
                    self.logger.info(f"Fixing always_hosts for {circle_id}: {circle_data.get('always_hosts', 'None')} → {always_hosts}")
                    circle_data['always_hosts'] = always_hosts
                
                if 'sometimes_hosts' not in circle_data or circle_data['sometimes_hosts'] != sometimes_hosts:
                    self.logger.info(f"Fixing sometimes_hosts for {circle_id}: {circle_data.get('sometimes_hosts', 'None')} → {sometimes_hosts}")
                    circle_data['sometimes_hosts'] = sometimes_hosts
                
            self.logger.info(f"Fixed metadata discrepancies for circle {circle_id}")
    
    def get_all_circles_enhanced(self, include_inactive=False):
        """
        Get all circle data with enhanced metadata for display.
        
        Args:
            include_inactive: Whether to include circles that have been replaced by splits
            
        Returns:
            List of dictionaries with enhanced circle data
        """
        # Extract circle data with detailed logging
        circle_data = []
        total_circles = len(self.circles)
        inactive_circles = 0
        active_circles = 0
        split_circles = 0
        
        print(f"\n🔍 ENHANCED CIRCLE DATA RETRIEVAL: Processing {total_circles} circles")
        
        for circle_id, circle in self.circles.items():
            # Standardized inactive status checking
            is_inactive = (
                circle.get('is_active') is False or  # explicitly inactive
                circle.get('active') is False or     # legacy inactive flag
                circle.get('replaced_by_splits', False)  # original circles that were split
            )
            
            # Skip inactive circles if not including them
            if not include_inactive and is_inactive:
                inactive_circles += 1
                continue
            else:
                active_circles += 1
                if circle.get('is_split_circle', False):
                    split_circles += 1
            
            # Debug info for specific test circles
            if circle_id in ['IP-SHA-01', 'IP-NAP-01', 'IP-ATL-1']:
                print(f"🔍 CIRCLE STATUS CHECK - {circle_id}:")
                print(f"  replaced_by_splits: {circle.get('replaced_by_splits', False)}")
                print(f"  active: {circle.get('active', True)}")
                print(f"  is_active: {circle.get('is_active', True)}")
                print(f"  is_inactive (calculated): {is_inactive}")
                print(f"  is_split_circle: {circle.get('is_split_circle', False)}")
                if circle.get('is_split_circle', False):
                    print(f"  original_circle_id: {circle.get('original_circle_id', 'unknown')}")
                elif circle.get('replaced_by_splits', False):
                    print(f"  split_into: {circle.get('split_into', [])}")
                
                print(f"  members: {len(circle.get('members', []))} members")
                print(f"  member_count: {circle.get('member_count', 0)}")
            
            # Skip inactive circles unless requested to include them - we've already counted them
            # But we also need to check the replaced_by_splits flag
            if not include_inactive and circle.get('replaced_by_splits', False):
                # Debug information for specific circles
                if circle_id in ['IP-SHA-01', 'IP-NAP-01', 'IP-ATL-1']:
                    print(f"⚠️ Skipping original circle {circle_id} that has been replaced by splits")
                continue
                
            # Always include split circles regardless of their active status
            # They should never be marked as inactive or replaced_by_splits
            if circle.get('is_split_circle', False):
                # Debug information for specific circles
                if circle_id in ['IP-SHA-01-SPLIT-01-A', 'IP-SHA-01-SPLIT-01-B', 'IP-NAP-01-SPLIT-01-A', 'IP-NAP-01-SPLIT-01-B']:
                    print(f"✅ Including split circle {circle_id} from original {circle.get('original_circle_id', 'unknown')}")
                # No continue, we want to include these
                
            # Get the enhanced circle data - first verify the circle exists
            if circle_id in self.circles:
                enhanced_data = self.get_circle_data(circle_id)
                if enhanced_data:
                    # CRITICAL FIX: Ensure member_count is accurate using the members list length
                    if 'members' in enhanced_data and isinstance(enhanced_data['members'], list):
                        actual_member_count = len(enhanced_data['members'])
                        if enhanced_data.get('member_count', 0) != actual_member_count:
                            print(f"⚠️ Member count mismatch for {circle_id}: stored={enhanced_data.get('member_count', 0)}, actual={actual_member_count}")
                            enhanced_data['member_count'] = actual_member_count
                    
                    circle_data.append(enhanced_data)
        
        print(f"✅ Processed {total_circles} circles: {active_circles} active, {inactive_circles} inactive, {split_circles} split")
        print(f"✅ Returning {len(circle_data)} circles after filtering")
        
        return circle_data
    
    def get_circle_data(self, circle_id):
        """
        Get the data for a specific circle with enhanced metadata.
        
        Args:
            circle_id: The ID of the circle to get data for
            
        Returns:
            Dictionary with circle data or None if not found
        """
        if circle_id in self.circles:
            # Make a copy of the data
            circle_data = self.circles[circle_id].copy()
            
            # Ensure circle_id is included in the data
            circle_data['circle_id'] = circle_id
            
            # Add a split status field for UI display
            if circle_id in self.split_circles:
                # For split circles, add the split letter
                split_letter = circle_data.get('split_letter', circle_id[-1] if circle_id[-1].isalpha() else "")
                circle_data['split_status'] = f"Split {split_letter}"
            elif circle_data.get('replaced_by_splits', False):
                # For circles that were split into multiple
                circle_data['split_status'] = "Original (Split)"
            else:
                circle_data['split_status'] = ""
            
            # Perform dynamic recalculation of metrics when results_df is available
            if self.results_df is not None:
                # Ensure member_count matches members list
                if 'members' in circle_data and isinstance(circle_data['members'], list):
                    circle_data['member_count'] = len(circle_data['members'])
                
                # Recalculate host counts if we have member data
                if 'members' in circle_data and circle_data['members']:
                    # Pass circle_id for debugging info
                    always_hosts, sometimes_hosts = self._count_hosts_from_members(circle_data['members'], circle_id)
                    circle_data['always_hosts'] = always_hosts
                    circle_data['sometimes_hosts'] = sometimes_hosts
                
            return circle_data
        
        return None
    
    def get_manager_from_session_state(session_state):
        """
        Get the CircleMetadataManager from the session state.
        
        Args:
            session_state: The session state object
            
        Returns:
            CircleMetadataManager or None
        """
        if hasattr(session_state, 'circle_metadata_manager') and session_state.circle_metadata_manager:
            return session_state.circle_metadata_manager
        return None
            
    def validate_max_additions(self) -> None:
        """Ensure max_additions is consistent with actual new members"""
        inconsistencies = 0
        corrections = 0
        target_circles = ['IP-BOS-04', 'IP-BOS-05']  # Specific circles we're troubleshooting
        
        print("\n🔍 MAX ADDITIONS VALIDATION: Checking consistency across all circles")
        
        # EXTENSIVE DIAGNOSTICS: Count occurrences of various max_additions values
        max_add_values = {}
        target_circle_data = {}
        all_boston_circles = []
        
        # First pass - collect stats
        for circle_id, circle in self.circles.items():
            max_add = circle.get('max_additions', 0)
            max_add_values[max_add] = max_add_values.get(max_add, 0) + 1
            
            # Track BOS circles specifically
            if circle_id.startswith('IP-BOS-'):
                all_boston_circles.append(circle_id)
                
            # Save target circle data for later analysis
            if circle_id in target_circles:
                target_circle_data[circle_id] = circle.copy()
        
        # Print distribution of max_additions values
        print(f"\n🔍 MAX ADDITIONS DISTRIBUTION ACROSS ALL CIRCLES:")
        print(f"  Total circles: {len(self.circles)}")
        print(f"  Distribution of max_additions values:")
        for max_add, count in sorted(max_add_values.items()):
            print(f"    max_additions={max_add}: {count} circles ({(count/len(self.circles))*100:.1f}%)")
        
        # Special analysis of Boston circles
        print(f"\n🔍 BOSTON CIRCLES ANALYSIS:")
        print(f"  Total Boston circles: {len(all_boston_circles)}")
        print(f"  Boston circle IDs: {sorted(all_boston_circles)}")
        
        # Fetch values from optimizer directly if available
        if hasattr(self, 'optimizer_circles') and self.optimizer_circles:
            print(f"\n🔍 COMPARING WITH ORIGINAL OPTIMIZER VALUES:")
            optimizer_data = {}
            for c in self.optimizer_circles:
                if isinstance(c, dict) and 'circle_id' in c:
                    c_id = c['circle_id']
                    optimizer_data[c_id] = {
                        'max_additions': c.get('max_additions', 'Not found'),
                        'always_hosts': c.get('always_hosts', 'Not found'),
                        'sometimes_hosts': c.get('sometimes_hosts', 'Not found')
                    }
            
            # Check our target circles
            for circle_id in target_circles:
                print(f"  TARGET CIRCLE {circle_id} COMPARISON:")
                if circle_id in optimizer_data:
                    opt_data = optimizer_data[circle_id]
                    print(f"    OPTIMIZER: max_add={opt_data['max_additions']}, always_hosts={opt_data['always_hosts']}, sometimes_hosts={opt_data['sometimes_hosts']}")
                    if circle_id in self.circles:
                        current = self.circles[circle_id]
                        print(f"    CURRENT:   max_add={current.get('max_additions', 'N/A')}, always_hosts={current.get('always_hosts', 'N/A')}, sometimes_hosts={current.get('sometimes_hosts', 'N/A')}")
                        
                        # Check for discrepancies
                        if current.get('max_additions', 0) != opt_data['max_additions'] and opt_data['max_additions'] != 'Not found':
                            print(f"    ⚠️ DISCREPANCY IN MAX_ADDITIONS: {current.get('max_additions', 0)} vs {opt_data['max_additions']}")
                        if current.get('always_hosts', 0) != opt_data['always_hosts'] and opt_data['always_hosts'] != 'Not found':
                            print(f"    ⚠️ DISCREPANCY IN ALWAYS_HOSTS: {current.get('always_hosts', 0)} vs {opt_data['always_hosts']}")
                        if current.get('sometimes_hosts', 0) != opt_data['sometimes_hosts'] and opt_data['sometimes_hosts'] != 'Not found':
                            print(f"    ⚠️ DISCREPANCY IN SOMETIMES_HOSTS: {current.get('sometimes_hosts', 0)} vs {opt_data['sometimes_hosts']}")
                else:
                    print(f"    Not found in optimizer data")
        
        # Second pass - validate and correct issues
        for circle_id, circle in self.circles.items():
            is_target = circle_id in target_circles
            
            # Special debug for target circles
            if is_target:
                print(f"\n🔍 DETAILED MAX ADDITIONS CHECK FOR {circle_id}:")
                print(f"  Circle data summary:")
                print(f"    member_count: {circle.get('member_count', 'N/A')}")
                print(f"    continuing_members: {circle.get('continuing_members', 'N/A')}")
                print(f"    new_members: {circle.get('new_members', 'N/A')}")
                print(f"    max_additions: {circle.get('max_additions', 'N/A')}")
                print(f"    always_hosts: {circle.get('always_hosts', 'N/A')}")
                print(f"    sometimes_hosts: {circle.get('sometimes_hosts', 'N/A')}")
                print(f"    full data: {circle}")
            
            if 'new_members' in circle and 'max_additions' in circle:
                new_members = circle.get('new_members', 0)
                max_additions = circle.get('max_additions', 0)
                
                # No special handling for Boston circles - rely on data-driven approach
                
                # No special handling - rely only on data-driven approach
                
                # If there are more new members than allowed, flag this
                if new_members > max_additions:
                    inconsistencies += 1
                    msg = f"⚠️ INCONSISTENCY: Circle {circle_id} has {new_members} new members but max_additions={max_additions}"
                    print(msg)
                    self.logger.warning(msg)
                    
                    # Correct max_additions to match reality if instructed
                    # In this implementation, we choose to update max_additions to match actual new members
                    circle['max_additions'] = new_members
                    corrections += 1
                    success_msg = f"✅ FIXED: Updated max_additions for {circle_id} to {new_members} to match actual new members"
                    print(success_msg)
                    self.logger.info(success_msg)
                    
                # Also look for suspiciously low max_additions
                elif max_additions == 0 and new_members == 0 and circle.get('member_count', 0) < 8:
                    # This might be a circle that should actually allow additions but doesn't
                    print(f"  🔎 SUSPICIOUS: Circle {circle_id} has {circle.get('member_count', 0)} members but max_additions=0")
                    # No automatic correction here, just flagging for attention
        
        # Now specially handle our target circles based on the screenshot evidence
        # NOTE: This is a temporary measure to fix known issues with specific circles
        # These fixes are based on the screenshot evidence shown by the user
        
        # No special handling for test circles
        # This ensures we rely solely on data-driven approaches
        print("  📊 Using pure data-driven approach for all circles - no special handling")
        
        summary = f"Found {inconsistencies} max_additions inconsistencies, applied {corrections} corrections"
        print(summary)
        self.logger.info(summary)
    
    def get_all_circles(self, include_inactive=False) -> List[Dict[str, Any]]:
        """
        Get all circle data as a list of dictionaries with dynamic recalculation
        
        This method is maintained for backward compatibility, but uses
        the enhanced implementation internally.
        
        Args:
            include_inactive: Whether to include circles that have been replaced by splits
            
        Returns:
            List of dictionaries with circle data
        """
        # Use the enhanced implementation with explicit parameter
        circles = self.get_all_circles_enhanced(include_inactive)
        
        # Add additional logging for debugging
        active_count = len([c for c in circles if not c.get('replaced_by_splits', False)])
        inactive_count = len([c for c in circles if c.get('replaced_by_splits', False)])
        self.logger.debug(f"get_all_circles returning {len(circles)} circles: {active_count} active, {inactive_count} inactive")
        
        return circles
    
    # This method has been merged with the new get_circle_data implementation above
    
    # This is now handled by the enhanced get_circles_dataframe method
    
    def update_circle(self, circle_id: str, **kwargs) -> None:
        """Update specific fields for a circle"""
        if circle_id in self.circles:
            self.circles[circle_id].update(kwargs)
            self.logger.debug(f"Updated circle {circle_id} with {kwargs}")
        else:
            self.logger.warning(f"Attempted to update non-existent circle {circle_id}")
    
    def update_all_circles(self, updated_circles: List[Dict[str, Any]]) -> None:
        """Update multiple circles at once from a list of circle dictionaries"""
        updated_count = 0
        
        for circle in updated_circles:
            circle_id = circle.get('circle_id')
            if circle_id and circle_id in self.circles:
                self.circles[circle_id].update(circle)
                updated_count += 1
        
        self.logger.info(f"Updated {updated_count} circles from batch update")
    
    def get_circles_by_region(self, region: str) -> List[Dict[str, Any]]:
        """Get all circles for a specific region"""
        return [c for c in self.circles.values() if c.get('region') == region]
    
    def get_continuing_circles(self) -> List[Dict[str, Any]]:
        """Get all continuing circles (not new)"""
        return [c for c in self.circles.values() if c.get('is_existing', False)]
    
    def get_new_circles(self) -> List[Dict[str, Any]]:
        """Get all new circles"""
        return [c for c in self.circles.values() if c.get('is_new_circle', False)]
    
    def get_circle_members(self, circle_id: str) -> List[str]:
        """
        Get member IDs for a specific circle
        
        Args:
            circle_id: The ID of the circle to get members for
            
        Returns:
            List of member IDs or empty list if circle not found
        """
        # Add special circle debugging for our test circles
        test_circle_ids = ['IP-BOS-04', 'IP-BOS-05', 'IP-ATL-1', 'IP-NAP-01', 'IP-SHA-01']
        is_test_circle = circle_id in test_circle_ids
        
        if is_test_circle:
            print(f"\n🔍🔍🔍 CIRCLE MEMBERS DEBUG FOR {circle_id} 🔍🔍🔍")
        
        if not circle_id:
            # Handle case where circle_id is None or empty
            self.logger.warning(f"Attempted to get members for invalid circle_id: '{circle_id}'")
            return []
        
        # First try using ParticipantDataManager if available
        if hasattr(self, 'participant_manager') and self.participant_manager is not None:
            try:
                # Check if the participant manager knows about this circle
                circle_members = self.participant_manager.get_circle_members(circle_id)
                
                if circle_members and len(circle_members) > 0:
                    if is_test_circle:
                        print(f"  ✅ Found {len(circle_members)} members via ParticipantDataManager for circle {circle_id}")
                        print(f"  ✅ Members: {circle_members}")
                    
                    self.logger.info(f"Retrieved {len(circle_members)} members from ParticipantDataManager for circle {circle_id}")
                    return circle_members
                
                if is_test_circle:
                    print(f"  ⚠️ ParticipantDataManager returned no members for circle {circle_id}, falling back to metadata")
            except Exception as e:
                if is_test_circle:
                    print(f"  ⚠️ Error getting members from ParticipantDataManager: {str(e)}")
                self.logger.warning(f"Error getting members from ParticipantDataManager: {str(e)}")
        elif is_test_circle:
            print(f"  ⚠️ No ParticipantDataManager available for {circle_id}")
            
        # Fall back to traditional approach using circle metadata
        circle = self.get_circle_data(circle_id)
        if circle is None:
            if is_test_circle:
                print(f"  ⚠️ Circle {circle_id} not found in metadata manager")
            self.logger.warning(f"Circle {circle_id} not found in metadata manager")
            return []
            
        # Get members with fallback to empty list
        members = circle.get('members', [])
        
        # Ensure members is a list and filter out invalid values
        member_list = self._ensure_list(members)
        
        if is_test_circle:
            print(f"  ✅ Found {len(member_list)} members via circle metadata for circle {circle_id}")
            print(f"  ✅ Members: {member_list}")
        
        # Log details for debugging
        self.logger.info(f"Retrieved {len(member_list)} members from circle metadata for circle {circle_id}")
        
        return member_list
    
    def get_circle_member_data(self, circle_id: str) -> pd.DataFrame:
        """Get detailed data for all members of a circle"""
        try:
            member_ids = self.get_circle_members(circle_id)
            
            if not member_ids or self.results_df is None:
                return pd.DataFrame()
            
            # Return subset of results_df for the specified members
            return self.results_df[self.results_df['Encoded ID'].isin(member_ids)]
        except Exception as e:
            print(f"⚠️ Error getting member data for circle {circle_id}: {str(e)}")
            return pd.DataFrame()
    
    def _ensure_list(self, value: Any) -> List:
        """Ensure a value is a list using standardized normalization"""
        from utils.data_standardization import normalize_member_list
        from utils.feature_flags import get_flag
        
        # If standardized member lists are enabled, use the normalization function
        if get_flag('use_standardized_member_lists'):
            return normalize_member_list(value)
        
        # Legacy implementation for backward compatibility
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            # Handle string representation of list
            if value.startswith('[') and value.endswith(']'):
                try:
                    import ast
                    return ast.literal_eval(value)
                except Exception as e:
                    self.logger.warning(f"Failed to parse string as list: {value}. Error: {str(e)}")
                    return []
            else:
                return [value]  # Single string item
        elif pd.isna(value):
            return []
        else:
            return [value]  # Single non-string item
    
    def export_to_session_state(self, state) -> None:
        """Save circle data to session state"""
        # Convert to DataFrame for storage in session_state
        circles_df = self.get_circles_dataframe()
        
        # Store in session_state
        state.matched_circles = circles_df
        self.logger.info(f"Exported {len(circles_df)} circles to session state")


# Helper functions for compatibility with existing code
def get_manager_from_session_state(state):
    """Get the circle manager from session state or create a new one 
    (this function is kept for backward compatibility, use CircleMetadataManager.get_manager_from_session_state instead)
    """
    return CircleMetadataManager.get_manager_from_session_state(state)


def initialize_or_update_manager(state, optimizer_circles=None, results_df=None):
    """Initialize or update the circle metadata manager in session state"""
    logger = logging.getLogger('circle_metadata_manager')
    
    # Debug input parameters
    print(f"\n🔄 INITIALIZING METADATA MANAGER: Input validation")
    print(f"  optimizer_circles type: {type(optimizer_circles)}")
    print(f"  results_df type: {type(results_df)}")
    
    # ENHANCED VALIDATION: Check for None and add detailed diagnostics
    if optimizer_circles is None:
        print(f"  ⚠️ WARNING: optimizer_circles is None. Will attempt to use matched_circles from session state.")
    elif len(str(optimizer_circles)) < 100:
        # Only print the full value if it's reasonably short
        print(f"  optimizer_circles value: {optimizer_circles}")
    
    # Validate optimizer_circles before using
    if optimizer_circles is not None:
        if isinstance(optimizer_circles, str):
            print(f"  ⚠️ WARNING: optimizer_circles is a string: '{optimizer_circles}'")
            # Handle string values specially
            try:
                if optimizer_circles.startswith('[') and optimizer_circles.endswith(']'):
                    # Try to parse as a list representation
                    import ast
                    optimizer_circles = ast.literal_eval(optimizer_circles)
                    print(f"  ⚙️ Successfully parsed string as a list with {len(optimizer_circles)} items")
                else:
                    # Treat as a single item
                    optimizer_circles = [optimizer_circles]
                    print(f"  ⚙️ Treating string as a single item list: {optimizer_circles}")
            except Exception as e:
                print(f"  ⚠️ ERROR parsing string optimizer_circles: {str(e)}")
                # Create an empty list as a fallback
                optimizer_circles = []
        elif not isinstance(optimizer_circles, (list, pd.DataFrame, dict)):
            print(f"  ⚠️ WARNING: optimizer_circles is {type(optimizer_circles)}, not a list/DataFrame/dict")
            try:
                # Try to convert to list as a last resort
                optimizer_circles = list(optimizer_circles) if hasattr(optimizer_circles, '__iter__') else [optimizer_circles]
                print(f"  ⚙️ Converted to list with {len(optimizer_circles)} items")
            except Exception as e:
                print(f"  ⚠️ ERROR converting to list: {str(e)}")
                # Create a single-item list with the original value
                optimizer_circles = [optimizer_circles]
    
    # Check if we have the necessary data
    if optimizer_circles is None and 'matched_circles' in state:
        # Try to use existing matched_circles
        print(f"  ⚙️ Using matched_circles from session state as fallback")
        optimizer_circles = state.matched_circles
        
        # Debug the matched_circles content
        print(f"  matched_circles type: {type(optimizer_circles)}")
        
        # Convert DataFrame to list of dicts if needed
        if hasattr(optimizer_circles, 'to_dict'):
            print(f"  ⚙️ Converting matched_circles DataFrame to dict records")
            optimizer_circles = optimizer_circles.to_dict('records')
    
    if results_df is None and 'results' in state:
        print(f"  ⚙️ Using results from session state as fallback")
        results_df = state.results
    
    # Additional validation
    if optimizer_circles is None:
        print(f"  ⚠️ ERROR: No circle data available. Cannot initialize manager.")
        return None
        
    if results_df is None:
        print(f"  ⚠️ WARNING: No results data available. Manager will have limited functionality.")
    
    # Check if optimizer_circles is an empty collection
    is_empty = False
    try:
        if isinstance(optimizer_circles, (list, pd.DataFrame)) and len(optimizer_circles) == 0:
            is_empty = True
    except Exception:
        # If len() raises an exception, treat as empty
        is_empty = True
    
    if is_empty:
        print(f"  ⚠️ ERROR: optimizer_circles is empty. Cannot initialize manager.")
        return None
    
    # Get or create participant data manager
    print(f"  🔄 Checking for ParticipantDataManager in session state")
    participant_manager = None
    if 'participant_data_manager' in state:
        participant_manager = state.participant_data_manager
        print(f"  ✅ Found existing ParticipantDataManager in session state")
    elif results_df is not None:
        from utils.participant_data_manager import ParticipantDataManager
        print(f"  🔄 Creating new ParticipantDataManager")
        participant_manager = ParticipantDataManager().initialize_from_dataframe(results_df)
        # Don't auto-store it since CircleMetadataManager will handle this
        print(f"  ✅ Created new ParticipantDataManager")
    else:
        print(f"  ⚠️ No results data available, cannot create ParticipantDataManager")
    
    # If we have circle data, initialize/update the manager
    try:
        # Create or retrieve manager
        if 'circle_manager' in state:
            print(f"  ⚙️ Updating existing CircleMetadataManager")
            manager = state.circle_manager
            # Make sure participant_manager is set
            if participant_manager is not None:
                manager.participant_manager = participant_manager
                print(f"  ✅ Connected ParticipantDataManager to existing CircleMetadataManager")
            manager.update_all_circles(optimizer_circles)
            manager.results_df = results_df
            manager.normalize_metadata()
            manager.validate_circles()
        else:
            print(f"  ⚙️ Creating new CircleMetadataManager")
            manager = CircleMetadataManager().initialize_from_optimizer(
                optimizer_circles, 
                results_df, 
                participant_manager
            )
            print(f"  ✅ Created new CircleMetadataManager with participant manager")
        
        # Store in session state
        state.circle_manager = manager
        
        # For backward compatibility, also update matched_circles
        try:
            circles_df = manager.get_circles_dataframe()
            state.matched_circles = circles_df
            print(f"  ✅ Successfully updated session state with {len(circles_df)} circles")
        except Exception as e:
            print(f"  ⚠️ Error updating matched_circles: {str(e)}")
            # Don't update matched_circles if there's an error to avoid corrupting it
        
        return manager
    except Exception as e:
        print(f"  ⚠️ Error in initialize_or_update_manager: {str(e)}")
        logger.error(f"Error in initialize_or_update_manager: {str(e)}")
        # Return existing manager if available as fallback
        if 'circle_manager' in state:
            return state.circle_manager
    
    return None
