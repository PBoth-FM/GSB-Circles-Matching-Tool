"""Centralized metadata management for region/subregion and meeting time information"""

import pandas as pd
import logging

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
