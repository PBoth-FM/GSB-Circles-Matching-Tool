"""Centralized metadata management for region/subregion and meeting time information"""

import pandas as pd
import logging
import sys

# Configure logging to ensure it shows in the console
logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Regional metadata mapping
REGION_SUBREGION_MAP = {
    'Peninsula': [
        'Atherton', 'Belmont', 'Burlingame', 'Emerald Hills', 'Foster City', 'Half Moon Bay',
        'Hillsborough', 'Menlo Park', 'Millbrae', 'Pacifica', 'Palo Alto', 'Portola Valley',
        'Redwood City', 'San Carlos', 'San Mateo', 'Woodside'
    ],
    'San Francisco': [
        'Bernal Heights', 'Buena Vista Park/Ashbury Heights', 'Castro', 'Cole Valley', 
        'Dogpatch/Potrero Hill', 'Excelsior', 'Financial District', 'Glen Park', 'Hayes Valley', 
        'Inner Richmond', 'Inner Sunset', 'Lower Haight', 'Marina', 'Mission', 'Mission Bay', 
        'Nob Hill', 'Noe Valley', 'North Beach', 'Outer Richmond', 'Outer Sunset', 'Pacific Heights', 
        'Pac Heights/Marina', 'Potrero Hill', 'Presidio', 'Presidio Heights', 'Russian Hill', 
        'SOMA', 'Tenderloin', 'Twin Peaks', 'Western Addition', 
        'Presidio/Marina/Pacific Heights'
    ],
    'Boston': [
        'Back Bay', 'Beacon Hill', 'Brighton', 'Brookline', 'Cambridge', 'Charlestown',
        'Downtown', 'East Boston', 'Fort Point', 'Jamaica Plain', 'Lexington', 'Newton',
        'North End', 'Somerville', 'South Boston', 'South End', 'Waltham', 'Watertown'
    ],
    'New York': [
        'Battery Park', 'Brooklyn', 'Brooklyn Heights', 'Chelsea', 'East Village', 'Financial District',
        'Flatiron', 'Gramercy', 'Greenwich Village', 'Hell\'s Kitchen', 'Long Island City',
        'Lower East Side', 'Midtown East', 'Midtown West', 'Murray Hill', 'NoHo', 'NoMad',
        'SoHo', 'Tribeca', 'Upper East Side', 'Upper West Side', 'West Village', 'Williamsburg'
    ],
    'East Bay': [
        'Alameda', 'Albany', 'Berkeley', 'Emeryville', 'Fremont', 'Hayward', 'Lafayette',
        'Livermore', 'Newark', 'Oakland', 'Orinda', 'Pleasanton', 'San Leandro', 'San Ramon',
        'Union City', 'Walnut Creek'
    ],
    'South Florida': [
        'Aventura', 'Boca Raton', 'Coconut Grove', 'Coral Gables', 'Delray Beach',
        'Doral', 'Downtown Miami', 'Fort Lauderdale', 'Hollywood', 'Key Biscayne',
        'Miami', 'Miami Beach', 'Palm Beach', 'Plantation', 'Pompano Beach',
        'South Beach', 'Sunny Isles', 'West Palm Beach', 'Weston'
    ]
}

# Fallback meeting times based on patterns observed in the data
FALLBACK_MEETING_TIMES = {
    'Peninsula': 'Varies (Evenings)',
    'San Francisco': 'Varies (Evenings)',
    'Boston': 'Tuesday (Evenings)',
    'New York': 'Varies (Evenings)',
    'East Bay': 'Sunday (Evenings)',
    'South Florida': 'Varies (Evenings)'
}

def fix_participant_metadata_in_results(results_df):
    """
    Apply metadata fixes directly to the participant results dataframe.
    This function fixes proposed_NEW_Subregion and proposed_NEW_DayTime values
    for all participants based on their assigned circles.
    
    Args:
        results_df: DataFrame with participant results including circle assignments
        
    Returns:
        DataFrame with fixed metadata values
    """
    print("\n🔄 APPLYING CENTRALIZED METADATA FIXES TO PARTICIPANT RESULTS")
    
    if not isinstance(results_df, pd.DataFrame) or results_df.empty:
        print("⚠️ Cannot fix empty or invalid results DataFrame")
        return results_df
    
    # Make a copy to avoid modifying the original
    fixed_results = results_df.copy()
    
    # Identify circles column
    circle_column = None
    for col in ['proposed_NEW_circles_id', 'assigned_circle', 'circle_id']:
        if col in fixed_results.columns:
            circle_column = col
            break
    
    if not circle_column:
        print("⚠️ Could not find circle assignment column in results!")
        return fixed_results
        
    print(f"Found circle assignments in column: {circle_column}")
    
    # Check if the metadata columns exist
    has_subregion = 'proposed_NEW_Subregion' in fixed_results.columns
    has_meeting_time = 'proposed_NEW_DayTime' in fixed_results.columns
    
    if not has_subregion and not has_meeting_time:
        print("⚠️ Results data does not contain metadata columns!")
        return fixed_results
    
    # Count current unknown values
    if has_subregion:
        unknown_subregions = fixed_results[fixed_results['proposed_NEW_Subregion'] == 'Unknown'].shape[0]
        print(f"Initial state: {unknown_subregions} participants with Unknown subregion")
    
    if has_meeting_time:
        unknown_times = fixed_results[fixed_results['proposed_NEW_DayTime'] == 'Unknown'].shape[0]
        print(f"Initial state: {unknown_times} participants with Unknown meeting time")
    
    # First approach: Build circle metadata by aggregating participant data by circle
    print("\nApproach 1: Build circle metadata from participant data")
    
    circle_metadata = {}
    
    # Get all matched participants (excluding UNMATCHED)
    matched_mask = (fixed_results[circle_column] != 'UNMATCHED') & fixed_results[circle_column].notna()
    matched_df = fixed_results[matched_mask]
    
    # Group by circle and extract valid metadata
    for circle_id, group in matched_df.groupby(circle_column):
        circle_metadata[circle_id] = {}
        
        # Find region for this circle
        if 'region' in group.columns:
            regions = group['region'].dropna().unique()
            if len(regions) > 0:
                circle_metadata[circle_id]['region'] = regions[0]
        
        # Try to find valid subregion data (non-Unknown values)
        if has_subregion:
            valid_subregions = group['proposed_NEW_Subregion'].dropna()
            valid_subregions = valid_subregions[valid_subregions != 'Unknown']
            
            if not valid_subregions.empty:
                circle_metadata[circle_id]['subregion'] = valid_subregions.iloc[0]
                print(f"✅ Found valid subregion for {circle_id}: '{valid_subregions.iloc[0]}'")
        
        # Try to find valid meeting time data (non-Unknown values)
        if has_meeting_time:
            valid_times = group['proposed_NEW_DayTime'].dropna()
            valid_times = valid_times[valid_times != 'Unknown']
            
            if not valid_times.empty:
                circle_metadata[circle_id]['meeting_time'] = valid_times.iloc[0]
                print(f"✅ Found valid meeting time for {circle_id}: '{valid_times.iloc[0]}'")
    
    print(f"Built metadata for {len(circle_metadata)} circles")
    
    # Apply the metadata fixes to all participants in each circle
    fixed_subregions = 0
    fixed_meeting_times = 0
    
    # Create a sample list to track examples of fixes
    subregion_fix_samples = []
    meeting_time_fix_samples = []
    
    # For each participant
    for i, row in fixed_results.iterrows():
        circle_id = row.get(circle_column)
        
        # Only fix participants who are assigned to a circle
        if pd.notna(circle_id) and circle_id != 'UNMATCHED' and circle_id in circle_metadata:
            region = row.get('region', '')
            circle_meta = circle_metadata[circle_id]
            
            # Fix subregion
            if has_subregion and row['proposed_NEW_Subregion'] == 'Unknown':
                # Method 1: Use shared circle metadata
                if 'subregion' in circle_meta:
                    fixed_results.at[i, 'proposed_NEW_Subregion'] = circle_meta['subregion']
                    fixed_subregions += 1
                    if len(subregion_fix_samples) < 5:
                        subregion_fix_samples.append((circle_id, circle_meta['subregion']))
                # Method 2: Use region-based defaults
                elif region and region in REGION_SUBREGION_MAP:
                    fixed_results.at[i, 'proposed_NEW_Subregion'] = region
                    fixed_subregions += 1
                    if len(subregion_fix_samples) < 5:
                        subregion_fix_samples.append((circle_id, region))
            
            # Fix meeting time
            if has_meeting_time and row['proposed_NEW_DayTime'] == 'Unknown':
                # Method 1: Use shared circle metadata
                if 'meeting_time' in circle_meta:
                    fixed_results.at[i, 'proposed_NEW_DayTime'] = circle_meta['meeting_time']
                    fixed_meeting_times += 1
                    if len(meeting_time_fix_samples) < 5:
                        meeting_time_fix_samples.append((circle_id, circle_meta['meeting_time']))
                # Method 2: Use region-based defaults
                elif region and region in FALLBACK_MEETING_TIMES:
                    fixed_results.at[i, 'proposed_NEW_DayTime'] = FALLBACK_MEETING_TIMES[region]
                    fixed_meeting_times += 1
                    if len(meeting_time_fix_samples) < 5:
                        meeting_time_fix_samples.append((circle_id, FALLBACK_MEETING_TIMES[region]))
    
    # Report results
    print(f"\n✅ Fixed {fixed_subregions} participant subregion values")
    if subregion_fix_samples:
        print(f"Sample subregion fixes (circle_id: new_value): {subregion_fix_samples}")
        
    print(f"✅ Fixed {fixed_meeting_times} participant meeting time values")
    if meeting_time_fix_samples:
        print(f"Sample meeting time fixes (circle_id: new_value): {meeting_time_fix_samples}")
    
    # Final verification
    if has_subregion:
        remaining = fixed_results[fixed_results['proposed_NEW_Subregion'] == 'Unknown'].shape[0]
        print(f"Final check: {remaining} participants still have Unknown subregion values")
        
    if has_meeting_time:
        remaining = fixed_results[fixed_results['proposed_NEW_DayTime'] == 'Unknown'].shape[0]
        print(f"Final check: {remaining} participants still have Unknown meeting time values")
    
    return fixed_results

def debug_circle_metadata(circle_df):
    """Debug circle metadata to identify issues"""
    if not isinstance(circle_df, pd.DataFrame) or circle_df.empty:
        logging.warning("Cannot debug empty or invalid circle DataFrame")
        return
    
    logging.info(f"\nCircle Metadata Debug Report")
    logging.info(f"Total circles: {len(circle_df)}")
    
    # Check subregion data
    if 'subregion' in circle_df.columns:
        unknown_count = circle_df[circle_df['subregion'] == 'Unknown'].shape[0]
        logging.info(f"Circles with 'Unknown' subregion: {unknown_count} ({unknown_count/len(circle_df)*100:.1f}%)")
        
        # Sample unknown subregions by region
        if unknown_count > 0:
            region_counts = circle_df[circle_df['subregion'] == 'Unknown']['region'].value_counts()
            logging.info(f"Regions with unknown subregions:")
            for region, count in region_counts.items():
                logging.info(f"  - {region}: {count} circles")
                # Show a few examples from each region
                examples = circle_df[(circle_df['region'] == region) & 
                                    (circle_df['subregion'] == 'Unknown')]['circle_id'].tolist()[:3]
                logging.info(f"    Examples: {examples}")
    
    # Check meeting time data
    if 'meeting_time' in circle_df.columns:
        unknown_count = circle_df[circle_df['meeting_time'] == 'Unknown'].shape[0]
        logging.info(f"Circles with 'Unknown' meeting time: {unknown_count} ({unknown_count/len(circle_df)*100:.1f}%)")
        
        # Sample unknown meeting times by region
        if unknown_count > 0:
            region_counts = circle_df[circle_df['meeting_time'] == 'Unknown']['region'].value_counts()
            logging.info(f"Regions with unknown meeting times:")
            for region, count in region_counts.items():
                logging.info(f"  - {region}: {count} circles")
                # Show a few examples from each region
                examples = circle_df[(circle_df['region'] == region) & 
                                    (circle_df['meeting_time'] == 'Unknown')]['circle_id'].tolist()[:3]
                logging.info(f"    Examples: {examples}")
    
    return

def fill_circle_metadata(circle_df, results_df):
    """Fill in missing circle metadata"""
    if not isinstance(circle_df, pd.DataFrame) or circle_df.empty:
        logging.warning("Cannot process empty or invalid circle DataFrame")
        return circle_df
    
    logging.info(f"\nApplying comprehensive metadata fixes")
    logging.info(f"Starting with {len(circle_df)} circles")
    
    # Track changes
    subregion_fixed = 0
    meeting_time_fixed = 0
    
    # Make a copy to avoid modifying the original
    fixed_df = circle_df.copy()
    
    # First approach: Use participant data from results_df to fill missing values
    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        logging.info("Method 1: Using participant data to fill missing values")
        
        # Check for the circles column
        circle_column = None
        for col in ['proposed_NEW_circles_id', 'assigned_circle', 'circle_id']:
            if col in results_df.columns:
                circle_column = col
                break
        
        if circle_column:
            # For each circle with missing data
            for i, row in fixed_df.iterrows():
                circle_id = row['circle_id']
                
                # Check if this circle needs fixes
                needs_subregion = row.get('subregion', '') == 'Unknown'
                needs_meeting_time = row.get('meeting_time', '') == 'Unknown'
                
                if needs_subregion or needs_meeting_time:
                    # Find members of this circle
                    circle_members = results_df[results_df[circle_column] == circle_id]
                    
                    if not circle_members.empty:
                        # Try to fix subregion
                        if needs_subregion:
                            # Check various possible subregion columns
                            subregion_cols = ['proposed_NEW_Subregion', 'Current_Subregion', 'Current Subregion',
                                             'Subregion1', 'Subregion2', 'Subregion3', 'Preferred_Subregion']
                            
                            for col in subregion_cols:
                                if col in circle_members.columns:
                                    # Get unique non-null, non-unknown subregions
                                    values = circle_members[col].dropna()
                                    values = values[values != 'Unknown'].unique()
                                    
                                    if len(values) > 0:
                                        fixed_df.at[i, 'subregion'] = values[0]
                                        subregion_fixed += 1
                                        logging.info(f"Fixed subregion for {circle_id}: '{values[0]}' (from {col})")
                                        break
                        
                        # Try to fix meeting time
                        if needs_meeting_time:
                            # Check various possible meeting time columns
                            time_cols = ['proposed_NEW_DayTime', 'Current_Meeting_Time', 'Current Meeting Time',
                                        'Meeting_Day_Time', 'Preferred_Meeting_Time', 'DayTime1', 'DayTime2', 'DayTime3']
                            
                            for col in time_cols:
                                if col in circle_members.columns:
                                    # Get unique non-null, non-unknown meeting times
                                    values = circle_members[col].dropna()
                                    values = values[values != 'Unknown'].unique()
                                    
                                    if len(values) > 0:
                                        fixed_df.at[i, 'meeting_time'] = values[0]
                                        meeting_time_fixed += 1
                                        logging.info(f"Fixed meeting time for {circle_id}: '{values[0]}' (from {col})")
                                        break
    
    # Second approach: Use region-based defaults for any remaining unknowns
    logging.info("Method 2: Using region-based defaults for remaining unknowns")
    
    for i, row in fixed_df.iterrows():
        region = row.get('region', '')
        circle_id = row.get('circle_id', '')
        
        # Fix subregion
        if row.get('subregion', '') == 'Unknown' and region in REGION_SUBREGION_MAP:
            # Use the region name as a default subregion
            fixed_df.at[i, 'subregion'] = region
            subregion_fixed += 1
            logging.info(f"Applied region-based default subregion for {circle_id}: '{region}'")
        
        # Fix meeting time
        if row.get('meeting_time', '') == 'Unknown' and region in FALLBACK_MEETING_TIMES:
            fallback = FALLBACK_MEETING_TIMES[region]
            fixed_df.at[i, 'meeting_time'] = fallback
            meeting_time_fixed += 1
            logging.info(f"Applied region-based default meeting time for {circle_id}: '{fallback}'")
    
    # Report results
    logging.info(f"Metadata fix summary:")
    logging.info(f"  - Fixed {subregion_fixed} circles with unknown subregions")
    logging.info(f"  - Fixed {meeting_time_fixed} circles with unknown meeting times")
    
    return fixed_df

def extract_column_values(df, column_patterns, default_value='Unknown'):
    """Extract values from columns matching specific patterns"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return default_value
    
    # Function to check if a column contains the pattern
    def contains_pattern(col_name, patterns):
        return any(pattern.lower() in col_name.lower() for pattern in patterns)
    
    # Find columns matching the patterns
    matching_cols = [col for col in df.columns if contains_pattern(col, column_patterns)]
    
    # Extract values from matching columns
    for col in matching_cols:
        # Get unique non-null values
        values = df[col].dropna().unique()
        if len(values) > 0 and values[0] != default_value:
            return values[0]
    
    return default_value
