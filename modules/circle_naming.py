"""
Circle naming utilities to generate proper circle IDs based on member composition.
"""
import pandas as pd
import os
from modules.circle_reconstruction import load_subregion_normalization_table, normalize_subregion

def load_region_subregion_mapping():
    """
    Load the region/subregion code mapping from CSV.
    
    Returns:
        Dictionary with mapping data
    """
    mapping_file = 'attached_assets/Circles-RegionSubregionCodeMapping.csv'
    
    if not os.path.exists(mapping_file):
        print(f"⚠️ Warning: Region mapping file not found at {mapping_file}")
        return {}
    
    try:
        df = pd.read_csv(mapping_file)
        mapping = {}
        
        for _, row in df.iterrows():
            format_type = str(row['Format']).strip()
            region = str(row['Region']).strip()
            subregion = str(row['Subregion']).strip()
            region_code = str(row['Region Code']).strip()
            
            # Create a key for lookup
            key = (format_type, region, subregion)
            mapping[key] = region_code
            
        print(f"✅ Loaded {len(mapping)} region/subregion mappings")
        return mapping
        
    except Exception as e:
        print(f"❌ Error loading region mapping: {str(e)}")
        return {}

def determine_circle_format_and_region(members_data, results_df):
    """
    Analyze circle members to determine the correct Format (VO/IP) and Region.
    
    Args:
        members_data: List of member Encoded IDs in the circle
        results_df: DataFrame with participant data
        
    Returns:
        Tuple of (format_code, region, subregion, region_code)
    """
    if not members_data or len(members_data) == 0:
        return "IP", "Unknown", "Unknown", "UNKNOWN"
    
    # Get member data from results
    member_rows = results_df[results_df['Encoded ID'].isin(members_data)]
    
    if member_rows.empty:
        return "IP", "Unknown", "Unknown", "UNKNOWN"
    
    # Analyze Derived_Region to determine format and region
    derived_regions = member_rows['Derived_Region'].dropna().unique()
    
    if len(derived_regions) == 0:
        return "IP", "Unknown", "Unknown", "UNKNOWN"
    
    # Use the most common region
    region_counts = member_rows['Derived_Region'].value_counts()
    most_common_region = region_counts.index[0]
    
    # Determine format: VO if Virtual-Only, otherwise IP
    if most_common_region.startswith('Virtual-Only'):
        format_code = "VO"
        # Extract the actual region part after "Virtual-Only "
        region = most_common_region.replace('Virtual-Only ', '').strip()
    else:
        format_code = "IP"
        region = most_common_region
    
    # Get the most common subregion
    subregions = member_rows['proposed_NEW_Subregion'].dropna().unique()
    if len(subregions) > 0:
        subregion_counts = member_rows['proposed_NEW_Subregion'].value_counts()
        subregion = subregion_counts.index[0]
        # Normalize the subregion
        subregion = normalize_subregion(subregion)
    else:
        subregion = "Unknown"
    
    # Load the mapping to get region code
    mapping = load_region_subregion_mapping()
    
    # Try to find the region code
    region_code = "UNKNOWN"
    
    # Create lookup key
    lookup_key = (format_code.replace("VO", "Virtual").replace("IP", "In Person"), region, subregion)
    
    if lookup_key in mapping:
        region_code = mapping[lookup_key]
    else:
        # Try partial matches - sometimes the region names might not match exactly
        for (map_format, map_region, map_subregion), map_code in mapping.items():
            if (map_format.startswith("Virtual") and format_code == "VO") or (map_format.startswith("In Person") and format_code == "IP"):
                if map_region.lower() == region.lower():
                    region_code = map_code
                    break
    
    print(f"🔍 Circle naming analysis: Format={format_code}, Region={region}, Subregion={subregion}, Code={region_code}")
    
    return format_code, region, subregion, region_code

def generate_proper_circle_name(circle_id, members_data, results_df, existing_names=None):
    """
    Generate a proper circle name based on member composition.
    
    Args:
        circle_id: Current circle ID (may be IP-UNKNOWN-NEW-01)
        members_data: List of member Encoded IDs in the circle
        results_df: DataFrame with participant data
        existing_names: Set of existing circle names to avoid duplicates
        
    Returns:
        New proper circle ID
    """
    if existing_names is None:
        existing_names = set()
    
    # Analyze member composition
    format_code, region, subregion, region_code = determine_circle_format_and_region(members_data, results_df)
    
    # If we couldn't determine proper naming, keep the unknown format
    if region_code == "UNKNOWN":
        print(f"⚠️ Could not determine proper region code for circle {circle_id}")
        return circle_id
    
    # For Virtual circles, we need to extract the timezone part for the subregion code
    if format_code == "VO":
        # Extract timezone from subregion (e.g., "GMT-5 (Eastern...)" -> "GMT-5")
        if "GMT" in subregion:
            import re
            gmt_match = re.search(r'GMT[+-]?\d*(?::\d+)?', subregion)
            if gmt_match:
                timezone = gmt_match.group(0)
                # The region code for virtual should be something like AE-GMT+1 or AM-GMT-5
                if region_code.startswith('AE-') or region_code.startswith('AM-'):
                    subregion_code = region_code.split('-', 1)[1]  # Extract the GMT part
                else:
                    subregion_code = timezone
            else:
                subregion_code = region_code
        else:
            subregion_code = region_code
    else:
        # For In Person circles, use the region code directly
        subregion_code = region_code
    
    # Generate base name pattern
    if format_code == "VO":
        # Virtual format: VO-{Region}-{Timezone}-NEW-{Number}
        base_pattern = f"{format_code}-{region_code.split('-')[0]}-{subregion_code}-NEW"
    else:
        # In Person format: IP-{RegionCode}-NEW-{Number}
        base_pattern = f"{format_code}-{subregion_code}-NEW"
    
    # Find the next available number
    counter = 1
    while True:
        new_name = f"{base_pattern}-{str(counter).zfill(2)}"
        if new_name not in existing_names:
            break
        counter += 1
    
    print(f"🔄 Circle naming: {circle_id} → {new_name}")
    return new_name

def fix_circle_naming_post_processing(results_df, circles_df):
    """
    Fix circle naming in post-processing by analyzing member composition.
    
    Args:
        results_df: DataFrame with participant results
        circles_df: DataFrame with circle data
        
    Returns:
        Tuple of (updated_results_df, updated_circles_df, naming_changes)
    """
    print("\n🔄 POST-PROCESSING: Fixing circle naming based on member composition")
    
    # Track naming changes
    naming_changes = {}
    existing_names = set()
    
    # First, collect all existing proper names to avoid conflicts
    if 'proposed_NEW_circles_id' in results_df.columns:
        existing_proper_names = results_df['proposed_NEW_circles_id'].dropna().unique()
        for name in existing_proper_names:
            if not name.startswith('IP-UNKNOWN'):
                existing_names.add(name)
    
    # Find circles that need renaming (those with IP-UNKNOWN or similar patterns)
    circles_to_rename = results_df[
        (results_df['proposed_NEW_circles_id'].str.contains('UNKNOWN', na=False)) |
        (results_df['proposed_NEW_circles_id'].str.contains('IP-NEW-', na=False))
    ]['proposed_NEW_circles_id'].unique()
    
    print(f"Found {len(circles_to_rename)} circles needing proper naming")
    
    # Process each circle that needs renaming
    for old_circle_id in circles_to_rename:
        # Get members of this circle
        circle_members = results_df[results_df['proposed_NEW_circles_id'] == old_circle_id]
        member_ids = circle_members['Encoded ID'].tolist()
        
        if len(member_ids) > 0:
            # Generate proper name
            new_circle_id = generate_proper_circle_name(old_circle_id, member_ids, results_df, existing_names)
            
            if new_circle_id != old_circle_id:
                naming_changes[old_circle_id] = new_circle_id
                existing_names.add(new_circle_id)
                print(f"  ✅ Renamed: {old_circle_id} → {new_circle_id} ({len(member_ids)} members)")
    
    # Apply naming changes to results DataFrame
    updated_results_df = results_df.copy()
    for old_name, new_name in naming_changes.items():
        mask = updated_results_df['proposed_NEW_circles_id'] == old_name
        updated_results_df.loc[mask, 'proposed_NEW_circles_id'] = new_name
        
        # Also update related region and subregion data if needed
        updated_members = updated_results_df[updated_results_df['proposed_NEW_circles_id'] == new_name]
        if not updated_members.empty:
            # Update Derived_Region based on the new naming
            format_code, region, subregion, region_code = determine_circle_format_and_region(
                updated_members['Encoded ID'].tolist(), updated_results_df)
            
            # Update the region information for consistency
            if format_code == "VO":
                proper_derived_region = f"Virtual-Only {region}"
            else:
                proper_derived_region = region
                
            updated_results_df.loc[updated_results_df['proposed_NEW_circles_id'] == new_name, 'Derived_Region'] = proper_derived_region
    
    # Apply naming changes to circles DataFrame if provided
    updated_circles_df = circles_df.copy() if circles_df is not None else None
    if updated_circles_df is not None and 'circle_id' in updated_circles_df.columns:
        for old_name, new_name in naming_changes.items():
            mask = updated_circles_df['circle_id'] == old_name
            updated_circles_df.loc[mask, 'circle_id'] = new_name
    
    print(f"✅ Applied {len(naming_changes)} circle name changes")
    
    return updated_results_df, updated_circles_df, naming_changes