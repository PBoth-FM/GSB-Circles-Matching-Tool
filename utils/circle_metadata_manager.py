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
    
    def initialize_from_optimizer(self, optimizer_circles: List[Dict[str, Any]], 
                               results_df: pd.DataFrame) -> 'CircleMetadataManager':
        """
        Initialize circle metadata from optimizer results and store reference to results DataFrame
        
        Args:
            optimizer_circles: List of circle dictionaries from optimizer
            results_df: DataFrame with participant results
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Initializing CircleMetadataManager with {len(optimizer_circles)} circles")
        
        # Store reference to results DataFrame for member lookups
        self.results_df = results_df
        
        # Initialize circles dictionary from optimizer circles
        self.circles = {}
        for circle in optimizer_circles:
            circle_id = circle.get('circle_id')
            if circle_id:
                self.circles[circle_id] = circle.copy()  # Create a deep copy to avoid reference issues
            else:
                self.logger.warning(f"Found circle without circle_id: {circle}")
        
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
        
        for circle_id, circle in self.circles.items():
            # Count hosts using results_df as source of truth
            if self.results_df is not None and 'members' in circle and circle['members']:
                # Convert to list if needed
                members_list = self._ensure_list(circle['members'])
                
                # Count by analyzing each member
                always_hosts, sometimes_hosts = self._count_hosts_from_members(members_list)
                
                # Update counts if they differ from current values
                always_before = circle.get('always_hosts', 0)
                sometimes_before = circle.get('sometimes_hosts', 0)
                
                if always_before != always_hosts:
                    circle['always_hosts'] = always_hosts
                    always_fixed += 1
                    self.logger.debug(f"Fixed always_hosts for {circle_id}: {always_before} → {always_hosts}")
                
                if sometimes_before != sometimes_hosts:
                    circle['sometimes_hosts'] = sometimes_hosts
                    sometimes_fixed += 1
                    self.logger.debug(f"Fixed sometimes_hosts for {circle_id}: {sometimes_before} → {sometimes_hosts}")
        
        self.logger.info(f"Host normalization complete: Fixed {always_fixed} always_hosts and {sometimes_fixed} sometimes_hosts values")
    
    def _count_hosts_from_members(self, member_ids: List[str]) -> tuple:
        """Count always and sometimes hosts from member list"""
        always_hosts = 0
        sometimes_hosts = 0
        
        if self.results_df is None:
            return always_hosts, sometimes_hosts
        
        # Ensure Encoded ID column exists
        if 'Encoded ID' not in self.results_df.columns:
            self.logger.warning("Cannot count hosts: 'Encoded ID' column missing from results DataFrame")
            return always_hosts, sometimes_hosts
        
        # Host column may have different names, try to find it
        host_col = None
        for col in ['host', 'Host']: 
            if col in self.results_df.columns:
                host_col = col
                break
        
        if not host_col:
            self.logger.warning("Cannot count hosts: No host column found in results DataFrame")
            return always_hosts, sometimes_hosts
        
        # Count hosts from results DataFrame
        for member_id in member_ids:
            # Look up this member in results_df
            member_rows = self.results_df[self.results_df['Encoded ID'] == member_id]
            
            if not member_rows.empty:
                # Get host status
                host_status = member_rows.iloc[0][host_col]
                
                # Count based on status
                if host_status in ['Always', 'Always Host']:
                    always_hosts += 1
                elif host_status in ['Sometimes', 'Sometimes Host']:
                    sometimes_hosts += 1
        
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
        
        for circle_id, circle in self.circles.items():
            if 'new_members' in circle and 'max_additions' in circle:
                new_members = circle.get('new_members', 0)
                max_additions = circle.get('max_additions', 0)
                
                # If there are more new members than allowed, flag this
                if new_members > max_additions:
                    inconsistencies += 1
                    self.logger.warning(f"⚠️ INCONSISTENCY: Circle {circle_id} has {new_members} new members but max_additions={max_additions}")
                    
                    # Correct max_additions to match reality if instructed
                    # In this implementation, we choose to update max_additions to match actual new members
                    circle['max_additions'] = new_members
                    corrections += 1
                    self.logger.info(f"✅ FIXED: Updated max_additions for {circle_id} to {new_members} to match actual new members")
        
        self.logger.info(f"Found {inconsistencies} max_additions inconsistencies, applied {corrections} corrections")
    
    def get_all_circles(self) -> List[Dict[str, Any]]:
        """Get all circle data as a list of dictionaries"""
        return list(self.circles.values())
    
    def get_circle_data(self, circle_id: str) -> Dict[str, Any]:
        """Get data for a specific circle"""
        return self.circles.get(circle_id, {})
    
    def get_circles_dataframe(self) -> pd.DataFrame:
        """Get all circle data as a pandas DataFrame"""
        return pd.DataFrame(self.get_all_circles())
    
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
        """Ensure a value is a list"""
        if isinstance(value, list):
            return value
        elif isinstance(value, str):
            # Handle string representation of list
            if value.startswith('[') and value.endswith(']'):
                try:
                    import ast
                    return ast.literal_eval(value)
                except:
                    self.logger.warning(f"Failed to parse string as list: {value}")
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
    # Check if we have the necessary data
    if optimizer_circles is None and 'matched_circles' in state:
        # Try to use existing matched_circles
        optimizer_circles = state.matched_circles
        
        # Convert DataFrame to list of dicts if needed
        if hasattr(optimizer_circles, 'to_dict'):
            optimizer_circles = optimizer_circles.to_dict('records')
    
    if results_df is None and 'results' in state:
        results_df = state.results
    
    # If we have circle data, initialize/update the manager
    if optimizer_circles is not None and results_df is not None:
        # Create or retrieve manager
        if 'circle_manager' in state:
            manager = state.circle_manager
            manager.update_all_circles(optimizer_circles)
            manager.results_df = results_df
            manager.normalize_metadata()
            manager.validate_circles()
        else:
            manager = CircleMetadataManager().initialize_from_optimizer(optimizer_circles, results_df)
        
        # Store in session state
        state.circle_manager = manager
        
        # For backward compatibility, also update matched_circles
        circles_df = manager.get_circles_dataframe()
        state.matched_circles = circles_df
        
        return manager
    
    return None
