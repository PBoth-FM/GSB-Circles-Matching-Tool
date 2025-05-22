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
                               results_df) -> 'CircleMetadataManager':
        """
        Initialize circle metadata from optimizer results and store reference to results DataFrame
        
        Args:
            optimizer_circles: List of circle dictionaries from optimizer
            results_df: DataFrame with participant results
            
        Returns:
            Self for method chaining
        """
        # Handle invalid or empty input
        if optimizer_circles is None:
            self.logger.error("Cannot initialize with None optimizer_circles")
            return self
        
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
        self.normalize_subregion_values()  # Added subregion normalization
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
        test_circle_ids = ['IP-BOS-04', 'IP-BOS-05']
        is_test_circle = circle_id in test_circle_ids if circle_id else False
        
        # ENHANCED DEBUG: More descriptive debug information
        debug_prefix = f"[CIRCLE: {circle_id}]" if circle_id else "[UNKNOWN CIRCLE]"
        
        # SPECIAL DEBUG: Targeted diagnostics for test circles
        if is_test_circle:
            print(f"\n🔍🔍🔍 SPECIAL TEST CIRCLE HOST DEBUG FOR {circle_id} 🔍🔍🔍")
            print(f"  {debug_prefix} Found {len(member_ids)} members in member_ids list")
            if len(member_ids) <= 10:  # Only print full list if it's reasonably short
                print(f"  {debug_prefix} Member IDs: {member_ids}")
        
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
    
    def normalize_subregion_values(self) -> None:
        """
        Normalize all subregion values using the standardized normalization tables.
        This ensures consistent subregion naming throughout the application and CSV exports.
        """
        self.logger.info("Normalizing subregion values")
        
        # Import normalize_subregion function from circle_reconstruction
        try:
            from modules.circle_reconstruction import normalize_subregion, clear_normalization_cache
            
            # Clear the normalization cache to ensure fresh data
            clear_normalization_cache()
            print("\n🔄 SUBREGION NORMALIZATION: Normalizing all circle subregion values")
            
            # Track special problem regions for enhanced logging
            problem_subregions = ['Napa Valley', 'North Spokane', 'Unknown']
            normalized_count = 0
            problem_fixed = 0
            
            # Process each circle
            for circle_id, circle in self.circles.items():
                if 'subregion' in circle and circle['subregion']:
                    # Store original value for comparison
                    original_value = circle['subregion']
                    
                    # Normalize the subregion
                    normalized_value = normalize_subregion(original_value)
                    
                    # Update if different
                    if original_value != normalized_value:
                        circle['subregion'] = normalized_value
                        normalized_count += 1
                        
                        # Special logging for problem regions
                        if original_value in problem_subregions:
                            problem_fixed += 1
                            print(f"  ✅ Fixed problem subregion for {circle_id}: '{original_value}' → '{normalized_value}'")
            
            print(f"  Normalized {normalized_count} subregion values ({problem_fixed} were from known problem regions)")
            self.logger.info(f"Normalized {normalized_count} subregion values ({problem_fixed} from problem regions)")
            
        except Exception as e:
            print(f"  ⚠️ Error normalizing subregion values: {str(e)}")
            self.logger.error(f"Error normalizing subregion values: {str(e)}")
    
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
    
    def get_all_circles(self) -> List[Dict[str, Any]]:
        """Get all circle data as a list of dictionaries with dynamic recalculation"""
        updated_circles = []
        
        # Process each circle with dynamic recalculation
        for circle_id in self.circles.keys():
            updated_circle = self.get_circle_data(circle_id)
            updated_circles.append(updated_circle)
        
        return updated_circles
    
    def get_circle_data(self, circle_id: str) -> Dict[str, Any]:
        """Get data for a specific circle with dynamic recalculation of key metrics"""
        # Get base data
        base_data = self.circles.get(circle_id, {})
        
        # Only proceed with dynamic calculations if we have access to results_df
        if self.results_df is not None and circle_id is not None:
            # Always do a fresh count of member_count from the raw results DataFrame
            # This ensures accurate member counts regardless of what was stored
            try:
                # Calculate member count based on Encoded ID and circle ID
                if 'proposed_NEW_circles_id' in self.results_df.columns:
                    # For new assignments, check proposed_NEW_circles_id
                    members_in_circle = self.results_df[self.results_df['proposed_NEW_circles_id'] == circle_id]
                    member_count = len(members_in_circle)
                    
                    # Also check if any members have this as current circle
                    current_circle_cols = [col for col in self.results_df.columns if 'current' in col.lower() and 'circle' in col.lower()]
                    if current_circle_cols:
                        current_col = current_circle_cols[0]
                        # Handle the case where the column might contain non-string values
                        current_members = self.results_df[self.results_df[current_col].astype(str) == circle_id]
                        # Only count current members not already counted in proposed
                        if not members_in_circle.empty and not current_members.empty:
                            current_only = current_members[~current_members['Encoded ID'].isin(members_in_circle['Encoded ID'])]
                            member_count += len(current_only)
                    
                    # Print verification for test circles
                    if circle_id in ['IP-BOS-04', 'IP-BOS-05']:
                        print(f"\n🔍 REAL-TIME MEMBER COUNT FOR {circle_id}:")
                        print(f"  Direct calculated member count: {member_count}")
                        print(f"  Previously stored member count: {base_data.get('member_count', 'Not stored')}")
                        
                    # Update member_count in the returned data
                    base_data['member_count'] = member_count
                
                # Dynamic recalculation - no hardcoded values
                # Let Streamlit UI handle max_additions via normal processing
            except Exception as e:
                print(f"⚠️ Error during dynamic member count calculation for {circle_id}: {str(e)}")
        
        return base_data
    
    def get_circles_dataframe(self) -> pd.DataFrame:
        """Get all circle data as a pandas DataFrame, filtering out circles with zero members"""
        all_circles = self.get_all_circles()
        
        # Filter out circles with zero members
        filtered_circles = []
        zero_member_count = 0
        
        for circle in all_circles:
            member_count = circle.get('member_count', 0)
            if member_count > 0:
                filtered_circles.append(circle)
            else:
                zero_member_count += 1
        
        # Log the filtering for debugging
        if zero_member_count > 0:
            self.logger.info(f"Filtered out {zero_member_count} circles with zero members")
            print(f"🔧 CircleMetadataManager: Filtered out {zero_member_count} circles with zero members")
        
        return pd.DataFrame(filtered_circles)
    
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
        """Get member IDs for a specific circle"""
        circle = self.get_circle_data(circle_id)
        members = circle.get('members', [])
        return self._ensure_list(members)
    
    def get_circle_member_data(self, circle_id: str) -> pd.DataFrame:
        """Get detailed data for all members of a circle"""
        member_ids = self.get_circle_members(circle_id)
        
        if not member_ids or self.results_df is None:
            return pd.DataFrame()
        
        # Return subset of results_df for the specified members
        return self.results_df[self.results_df['Encoded ID'].isin(member_ids)]
    
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
    """Get the circle manager from session state or create a new one"""
    if 'circle_manager' in state:
        return state.circle_manager
    
    # Create a new manager if not found (as a fallback)
    logger = logging.getLogger('circle_metadata_manager')
    logger.warning("Creating new CircleMetadataManager because none was found in session state")
    return CircleMetadataManager()


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
    
    # If we have circle data, initialize/update the manager
    try:
        # Create or retrieve manager
        if 'circle_manager' in state:
            print(f"  ⚙️ Updating existing CircleMetadataManager")
            manager = state.circle_manager
            manager.update_all_circles(optimizer_circles)
            manager.results_df = results_df
            manager.normalize_metadata()
            manager.validate_circles()
        else:
            print(f"  ⚙️ Creating new CircleMetadataManager")
            manager = CircleMetadataManager().initialize_from_optimizer(optimizer_circles, results_df)
        
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
