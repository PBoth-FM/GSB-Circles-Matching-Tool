"""
Participant Data Manager module that provides a single source of truth for
participant information throughout the application.

This ensures consistent access to participant data across all components,
eliminating data synchronization issues and undefined variable errors.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Union, Optional, Set

class ParticipantDataManager:
    """
    Central manager for all participant data throughout the application.
    Provides a single source of truth for participant information.
    
    This class handles both continuing and new participants with consistent interfaces
    while providing specialized handling where needed based on participant type.
    """
    
    def __init__(self):
        """Initialize the ParticipantDataManager"""
        self.participants_df = None  # Main DataFrame containing all participant data
        self._initialized = False
        self.logger = self._setup_logger()
        self.participant_circle_map = {}  # Dictionary mapping participant IDs to circle IDs
        self.circle_participants_map = {}  # Dictionary mapping circle IDs to sets of participant IDs
        
    def _setup_logger(self):
        """Setup a logger for the participant data manager"""
        logger = logging.getLogger('participant_data_manager')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def initialize_from_dataframe(self, participants_df: pd.DataFrame) -> 'ParticipantDataManager':
        """
        Initialize participant data from a DataFrame
        
        Args:
            participants_df: DataFrame containing participant information
            
        Returns:
            Self for method chaining
        """
        # Handle invalid input
        if participants_df is None or not isinstance(participants_df, pd.DataFrame):
            self.logger.error("Cannot initialize with None or non-DataFrame participants_df")
            return self
        
        # Store reference to DataFrame
        self.participants_df = participants_df.copy()
        
        # Set initialized flag
        self._initialized = True
        
        # Find ID column
        id_col = self._get_id_column()
        if id_col is None:
            self.logger.error("Could not find participant ID column in data")
            return self
            
        # Find circle assignment columns
        circle_cols = self._get_circle_assignment_columns()
        if not circle_cols:
            self.logger.warning("Could not find circle assignment columns in data")
            
        # Build participant-circle and circle-participant maps
        self._build_relationship_maps(id_col, circle_cols)
        
        self.logger.info(f"Successfully initialized with {len(self.participants_df)} participants")
        return self
        
    def _get_id_column(self) -> Optional[str]:
        """
        Find the participant ID column in the DataFrame
        
        Returns:
            Column name or None if not found
        """
        id_column_candidates = ['Encoded ID', 'encoded_id', 'participant_id']
        for col in id_column_candidates:
            if col in self.participants_df.columns:
                self.logger.info(f"Using '{col}' as participant ID column")
                return col
        return None
    
    def _get_circle_assignment_columns(self) -> List[str]:
        """
        Find the circle assignment columns in the DataFrame
        
        Returns:
            List of column names for circle assignments
        """
        circle_columns = []
        
        # Look for current circle column (for continuing participants)
        current_circle_candidates = ['Current_Circle_ID', 'current_circle_id']
        for col in current_circle_candidates:
            if col in self.participants_df.columns:
                circle_columns.append(col)
                self.logger.info(f"Found column for continuing circles: '{col}'")
                break
                
        # Look for general/new assignment column
        assignment_candidates = ['assigned_circle', 'circle_id', 'Circle ID', 'proposed_NEW_circles_id']
        for col in assignment_candidates:
            if col in self.participants_df.columns:
                circle_columns.append(col)
                self.logger.info(f"Found column for circle assignments: '{col}'")
                break
                
        return circle_columns
    
    def _build_relationship_maps(self, id_col: str, circle_cols: List[str]) -> None:
        """
        Build maps between participants and circles
        
        Args:
            id_col: Column name for participant IDs
            circle_cols: List of column names for circle assignments
        """
        # Reset maps
        self.participant_circle_map = {}
        self.circle_participants_map = {}
        
        # Track stats for logging
        total_assignments = 0
        
        # Process each participant
        for idx, row in self.participants_df.iterrows():
            # Get participant ID
            participant_id = row[id_col]
            if pd.isna(participant_id) or participant_id == "":
                continue  # Skip participants without an ID
                
            # Convert to string for consistency
            participant_id = str(participant_id)
            
            # Initialize with no assignment
            self.participant_circle_map[participant_id] = None
            
            # Check each circle assignment column
            for circle_col in circle_cols:
                if circle_col in row and not pd.isna(row[circle_col]) and row[circle_col] != "":
                    circle_id = str(row[circle_col])
                    
                    # Update participant -> circle map
                    self.participant_circle_map[participant_id] = circle_id
                    
                    # Update circle -> participants map
                    if circle_id not in self.circle_participants_map:
                        self.circle_participants_map[circle_id] = set()
                    self.circle_participants_map[circle_id].add(participant_id)
                    
                    total_assignments += 1
                    break  # Stop after first valid assignment
        
        self.logger.info(f"Built relationship maps with {total_assignments} participant-circle assignments")
        self.logger.info(f"Found {len(self.circle_participants_map)} circles with assigned participants")
    
    def get_all_participants(self) -> pd.DataFrame:
        """
        Get all participants data
        
        Returns:
            DataFrame with all participant data
        """
        if not self._initialized:
            self.logger.warning("Attempting to get participants before initialization")
            return pd.DataFrame()
            
        return self.participants_df
    
    def get_participants_by_circle(self, circle_id: str) -> pd.DataFrame:
        """
        Get all participants in a specific circle
        
        Args:
            circle_id: ID of the circle
            
        Returns:
            DataFrame with participants in the circle
        """
        if not self._initialized:
            self.logger.warning("Attempting to get participants before initialization")
            return pd.DataFrame()
            
        if circle_id not in self.circle_participants_map:
            return pd.DataFrame()
            
        # Get participant IDs in this circle
        participant_ids = self.circle_participants_map[circle_id]
        
        # Find ID column
        id_col = self._get_id_column()
        if id_col is None:
            return pd.DataFrame()
            
        # Filter DataFrame to only these participants
        circle_participants = self.participants_df[self.participants_df[id_col].astype(str).isin(participant_ids)]
        
        return circle_participants
    
    def get_participant_ids_by_circle(self, circle_id: str) -> List[str]:
        """
        Get IDs of all participants in a specific circle
        
        Args:
            circle_id: ID of the circle
            
        Returns:
            List of participant IDs in the circle
        """
        if not self._initialized:
            self.logger.warning("Attempting to get participants before initialization")
            return []
            
        if circle_id not in self.circle_participants_map:
            return []
            
        return list(self.circle_participants_map[circle_id])
    
    def get_circle_members(self, circle_id: str) -> List[str]:
        """
        Get IDs of all members in a specific circle (alias for get_participant_ids_by_circle)
        
        Args:
            circle_id: ID of the circle
            
        Returns:
            List of member IDs in the circle
        """
        # This is an alias method for consistency with CircleMetadataManager
        return self.get_participant_ids_by_circle(circle_id)
    
    def get_circle_by_participant(self, participant_id: str) -> Optional[str]:
        """
        Get the circle ID for a specific participant
        
        Args:
            participant_id: ID of the participant
            
        Returns:
            Circle ID or None if not assigned
        """
        if not self._initialized:
            self.logger.warning("Attempting to get circle before initialization")
            return None
            
        # Convert to string for consistency
        participant_id = str(participant_id)
        
        return self.participant_circle_map.get(participant_id)
    
    def get_participant_by_id(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific participant by ID
        
        Args:
            participant_id: ID of the participant
            
        Returns:
            Dictionary with participant data or None if not found
        """
        if not self._initialized:
            self.logger.warning("Attempting to get participant before initialization")
            return None
            
        # Find ID column
        id_col = self._get_id_column()
        if id_col is None:
            return None
            
        # Convert to string for consistency
        participant_id = str(participant_id)
        
        # Find this participant
        matches = self.participants_df[self.participants_df[id_col].astype(str) == participant_id]
        
        if len(matches) == 0:
            return None
            
        # Return as dictionary
        return matches.iloc[0].to_dict()
    
    def get_participant_host_status(self, participant_id: str, debug_mode=False) -> str:
        """
        Get the host status for a specific participant
        
        Args:
            participant_id: ID of the participant
            debug_mode: If True, prints additional debug information
            
        Returns:
            Host status ("always", "sometimes", "never") or empty string if not found
        """
        participant_data = self.get_participant_by_id(participant_id)
        if not participant_data:
            if debug_mode:
                print(f"⚠️ Participant {participant_id} not found in data")
            return ""
        
        # Dictionary to map raw host values to standardized values
        # ENHANCED: Comprehensive mapping of all possible host values
        HOST_VALUE_MAP = {
            # ALWAYS mappings
            "always": "always",
            "always host": "always",
            "always_host": "always",
            "alwayshost": "always",
            "yes, always": "always",
            "yes-always": "always",
            "yes always": "always",
            "always willing": "always",
            "a": "always",
            "yes-a": "always",
            "yes (a)": "always",
            "yes": "always",
            "1": "always",
            "1.0": "always",
            # SOMETIMES mappings
            "sometimes": "sometimes",
            "sometimes host": "sometimes", 
            "sometimes_host": "sometimes",
            "sometimeshost": "sometimes",
            "yes, sometimes": "sometimes",
            "yes-sometimes": "sometimes",
            "yes sometimes": "sometimes",
            "sometimes willing": "sometimes",
            "s": "sometimes",
            "yes-s": "sometimes",
            "yes (s)": "sometimes",
            "maybe": "sometimes",
            "0.5": "sometimes",
            # NEVER mappings
            "n/a": "never",
            "na": "never",
            "never": "never",
            "never host": "never",
            "never_host": "never",
            "neverhost": "never",
            "not available": "never",
            "not willing": "never",
            "cannot host": "never",
            "n": "never",
            "no-n": "never",
            "no (n)": "never",
            "no": "never",
            "0": "never",
            "0.0": "never",
        }
        
        if debug_mode:
            print(f"🔍 DETAILED HOST DEBUG for participant {participant_id}")
            # Print all host-related fields in participant data
            host_related_columns = {}
            for col in participant_data:
                if 'host' in col.lower() or col.lower() in ['willing_to_host', 'hostingpreference']:
                    host_related_columns[col] = participant_data[col]
            if host_related_columns:
                print(f"  Host-related columns found: {host_related_columns}")
            
        # Try standardized host status first
        for col in ['host_status_standardized', 'Standardized Host', 'proposed_NEW_host_status']:
            if col in participant_data and not pd.isna(participant_data[col]):
                status = participant_data[col]
                if status:
                    normalized = str(status).lower().strip()
                    if debug_mode:
                        print(f"  🔍 Found standardized host status in '{col}': '{normalized}'")
                    # Map standardized statuses directly
                    if normalized in HOST_VALUE_MAP:
                        return HOST_VALUE_MAP[normalized]
                
        # Fall back to raw 'host' column (column 29) which is the main source in most datasets
        if 'host' in participant_data and not pd.isna(participant_data['host']):
            raw_host = str(participant_data['host']).lower().strip()
            if debug_mode:
                print(f"  🔍 Using raw 'host' value: '{raw_host}'")
            
            # Use our direct mapping for exact matches first
            if raw_host in HOST_VALUE_MAP:
                return HOST_VALUE_MAP[raw_host]
            
            # For partial matches or other variations
            if 'always' in raw_host:
                return 'always'
            elif 'sometimes' in raw_host:
                return 'sometimes'
            elif 'never' in raw_host or 'n/a' in raw_host or raw_host == '':
                return 'never'
                
            # Handle numeric values that might be cast to strings
            if raw_host.isdigit():
                if raw_host == '1': 
                    return 'always'
                elif raw_host == '0': 
                    return 'never'
                
            # Other potential patterns that might appear in the data
            # For floating point values like "1.0" or "0.5" converted to strings
            try:
                numeric_value = float(raw_host)
                if numeric_value == 1.0:
                    return 'always'
                elif numeric_value == 0.5:
                    return 'sometimes'
                elif numeric_value == 0.0:
                    return 'never'
            except ValueError:
                # Not a numeric value, continue to other checks
                pass
                
        # Try other host status columns with more specific naming
        for col in ['Host', 'HostingPreference', 'Host Status', 'willing_to_host', 
                   'Current Host Status', 'Future Host Status', 'Host_Status', 
                   'Can Host', 'Hosting', 'Proposed Host']:
            if col in participant_data and not pd.isna(participant_data[col]):
                status = str(participant_data[col]).lower().strip()
                if debug_mode:
                    print(f"  🔍 Checking host status in '{col}': '{status}'")
                
                # Try direct mapping first
                if status in HOST_VALUE_MAP:
                    return HOST_VALUE_MAP[status]
                
                # Then try partial matches
                if 'always' in status or status == 'yes' or status == '1':
                    return 'always'
                elif 'sometimes' in status or status == 'maybe' or status == '0.5':
                    return 'sometimes'
                elif 'never' in status or 'n/a' in status or status == 'no' or status == '0' or status == '':
                    return 'never'
                
                # Try numeric conversion as a last resort
                try:
                    numeric_value = float(status)
                    if numeric_value == 1.0:
                        return 'always'
                    elif numeric_value == 0.5:
                        return 'sometimes'
                    elif numeric_value == 0.0:
                        return 'never'
                except ValueError:
                    # Not a numeric value, continue to other checks
                    pass
        
        # Enhanced debugger for test cases
        if debug_mode:
            print(f"  🔍 DETAILED FALLBACK HOST DEBUG for participant {participant_id}")
            # Find all host-related columns in the data
            host_related_columns = {}
            for col in participant_data:
                if ('host' in col.lower() or 
                    'willing' in col.lower() or 
                    'can_host' in col.lower() or 
                    'preference' in col.lower()):
                    host_related_columns[col] = participant_data[col]
            
            if host_related_columns:
                print(f"  ⚠️ No conclusive host status despite finding these host-related columns:")
                for col, value in host_related_columns.items():
                    print(f"    - {col}: {value}")
            else:
                print(f"  ⚠️ NO host-related columns found in participant data!")
            
        # Final fallback - default to never
        if debug_mode:
            print(f"  ⚠️ No host status found, defaulting to 'never' for {participant_id}")
        return "never"
    
    def is_participant_co_leader(self, participant_id: str) -> bool:
        """
        Check if a participant is a co-leader
        
        Args:
            participant_id: ID of the participant
            
        Returns:
            True if the participant is a co-leader, False otherwise
        """
        participant_data = self.get_participant_by_id(participant_id)
        if not participant_data:
            return False
            
        # Check co-leader columns
        for col in ['co_leader', 'Co-Leader', 'CoLeader', 'Is Co-Leader']:
            if col in participant_data:
                value = participant_data[col]
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ['true', 'yes', 'y', '1']
                elif isinstance(value, (int, float)):
                    return value == 1
                
        return False
    
    def update_participant_circle(self, participant_id: str, circle_id: Optional[str]) -> bool:
        """
        Update the circle assignment for a participant
        
        Args:
            participant_id: ID of the participant
            circle_id: New circle ID or None to remove assignment
            
        Returns:
            True if successful, False otherwise
        """
        if not self._initialized:
            self.logger.warning("Attempting to update participant before initialization")
            return False
            
        # Find ID column
        id_col = self._get_id_column()
        if id_col is None:
            return False
            
        # Convert to string for consistency
        participant_id = str(participant_id)
        
        # Find assignment column
        circle_cols = self._get_circle_assignment_columns()
        if not circle_cols:
            return False
            
        # Select the first assignment column (general or new assignments)
        assignment_col = circle_cols[-1]  # Usually the last one is for new assignments
        
        # Find this participant
        matches = self.participants_df[self.participants_df[id_col].astype(str) == participant_id]
        
        if len(matches) == 0:
            return False
            
        # Get current circle assignment
        old_circle_id = self.participant_circle_map.get(participant_id)
        
        # Remove from old circle
        if old_circle_id and old_circle_id in self.circle_participants_map:
            self.circle_participants_map[old_circle_id].discard(participant_id)
            
        # Update DataFrame
        idx = matches.index[0]
        self.participants_df.at[idx, assignment_col] = circle_id
        
        # Update maps
        if circle_id:
            # Convert to string
            circle_id = str(circle_id)
            
            # Update participant -> circle map
            self.participant_circle_map[participant_id] = circle_id
            
            # Update circle -> participants map
            if circle_id not in self.circle_participants_map:
                self.circle_participants_map[circle_id] = set()
            self.circle_participants_map[circle_id].add(participant_id)
        else:
            # Remove assignment
            self.participant_circle_map[participant_id] = None
            
        return True
    
    def get_manager_from_session_state(session_state):
        """
        Get the ParticipantDataManager from the session state.
        
        Args:
            session_state: The session state object
            
        Returns:
            ParticipantDataManager or None
        """
        if hasattr(session_state, 'participant_data_manager') and session_state.participant_data_manager:
            return session_state.participant_data_manager
        return None
    
    def get_or_create_from_session_state(session_state, participants_df=None):
        """
        Get the ParticipantDataManager from session state or create a new one.
        
        Args:
            session_state: The session state object
            participants_df: Optional DataFrame to initialize with
            
        Returns:
            ParticipantDataManager instance
        """
        manager = ParticipantDataManager.get_manager_from_session_state(session_state)
        
        if not manager:
            manager = ParticipantDataManager()
            if participants_df is not None:
                manager.initialize_from_dataframe(participants_df)
            session_state.participant_data_manager = manager
            
        return manager