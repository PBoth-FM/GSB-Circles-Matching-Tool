import pandas as pd
import numpy as np
import re
import os

# Load normalization tables
def load_normalization_tables():
    """
    Load the region and subregion normalization tables
    
    Returns:
        Tuple of (region_mapping, subregion_mapping, region_code_mapping)
    """
    # First try to load from CSV files (for production)
    region_mapping = {}
    subregion_mapping = {}
    region_code_mapping = {}
    region_subregion_mapping = {}
    
    try:
        # Try to load region normalization
        if os.path.exists('attached_assets/Circles-RegionNormalization.csv'):
            print("✅ Using new Circles-RegionNormalization.csv file")
            region_df = pd.read_csv('attached_assets/Circles-RegionNormalization.csv')
            region_mapping = dict(zip(region_df['All unique variations'], region_df['Normalized Region']))
        elif os.path.exists('attached_assets/Appendix2-RegionNormalizationCodes.csv'):
            print("⚠️ Using legacy Appendix2-RegionNormalizationCodes.csv file")
            region_df = pd.read_csv('attached_assets/Appendix2-RegionNormalizationCodes.csv')
            region_mapping = dict(zip(region_df['All unique Region variations'], region_df['Normalized Region']))
            # Also create a mapping from normalized region names to region codes
            region_code_mapping = dict(zip(region_df['Normalized Region'], region_df['Region Code']))
    except Exception as e:
        print(f"Could not load region normalization table: {str(e)}")
        
    try:
        # Try to load subregion normalization
        if os.path.exists('attached_assets/Circles-SubregionNormalization.csv'):
            print("✅ Using new Circles-SubregionNormalization.csv file")
            subregion_df = pd.read_csv('attached_assets/Circles-SubregionNormalization.csv')
            subregion_mapping = dict(zip(subregion_df['All unique variations'], subregion_df['Normalized']))
        elif os.path.exists('attached_assets/Appendix1-SubregionNormalization.csv'):
            print("⚠️ Using legacy Appendix1-SubregionNormalization.csv file")
            subregion_df = pd.read_csv('attached_assets/Appendix1-SubregionNormalization.csv')
            subregion_mapping = dict(zip(subregion_df['All unique variations'], subregion_df['Normalized']))
    except Exception as e:
        print(f"Could not load subregion normalization table: {str(e)}")
    
    # Load Region-Subregion-Code mapping
    try:
        if os.path.exists('attached_assets/Circles-RegionSubregionCodeMapping.csv'):
            print("✅ Using new Circles-RegionSubregionCodeMapping.csv file")
            mapping_df = pd.read_csv('attached_assets/Circles-RegionSubregionCodeMapping.csv')
            # Create a more sophisticated region code mapping
            for _, row in mapping_df.iterrows():
                format_type = row['Format']
                region = row['Region']
                subregion = row['Subregion']
                region_code = row['Region Code']
                
                # For in-person circles, map the normalized region to region code
                if format_type == 'In Person':
                    if region not in region_code_mapping:
                        region_code_mapping[region] = region_code
                
                # Store a mapping of (normalized_region, normalized_subregion) to region_code
                # This is crucial for virtual circles where region code depends on subregion
                key = (region, subregion)
                region_subregion_mapping[key] = region_code
    except Exception as e:
        print(f"Could not load region-subregion-code mapping: {str(e)}")
    
    # If files not found, use hardcoded fallback (minimal version)
    if not region_mapping:
        region_mapping = {
            'South Florida': 'South Florida',
            'Boston': 'Boston',
            'New York': 'New York',
            'Washington DC': 'Washington DC',
            'Atlanta': 'Atlanta',
            'Chicago': 'Chicago',
            'Houston': 'Houston',
            'Austin': 'Austin',
            'San Francisco': 'San Francisco',
            'East Bay': 'East Bay',
            'Peninsula': 'Peninsula',
            'Marin': 'Marin County',
            'Marin County': 'Marin County',
            'Napa Sonoma': 'Napa-Sonoma',
            'Napa/Sonoma': 'Napa-Sonoma',
            'Napa-Sonoma': 'Napa-Sonoma',
            'Los Angeles': 'Los Angeles',
            'San Diego': 'San Diego',
            'Seattle': 'Seattle',
            'London': 'London',
            'Sao Paulo': 'Sao Paulo',
            'SÃ£o Paulo': 'Sao Paulo',
            'Mexico City': 'Mexico City',
            'Singapore': 'Singapore',
            'Shanghai': 'Shanghai',
            'Nairobi': 'Nairobi',
            'Virtual-Only Americas': 'Virtual-Only Americas',
            'Virtual-Only APAC+EMEA': 'Virtual-Only APAC+EMEA'
        }
    
    if not subregion_mapping:
        subregion_mapping = {
            'Miami': 'Miami',
            'Fort Lauderdale': 'Fort Lauderdale',
            'Palm Beach': 'Palm Beach',
            'West Palm Beach': 'West Palm Beach',
            'Boston': 'Boston',
            'Cambridge/Somerville': 'Cambridge/Somerville',
            'Brookline/Newton': 'Brookline/Newton',
            'North of Boston': 'North of Boston',
            'South of Boston': 'South of Boston',
            'West of Boston (e.g. Needham/Wellesley/Natick)': 'West of Boston (e.g. Needham/Wellesley/Natick)',
            'West Boston (e.g. Needham/Wellesley/Natick)': 'West of Boston (e.g. Needham/Wellesley/Natick)',
            'Northwest of Boston (e.g. Belmont/Lexington/Concord)': 'Northwest of Boston (e.g. Belmont/Lexington/Concord)',
            'San Francisco': 'San Francisco',
            'Pac Heights/Marina': 'Pac Heights/Marina',
            'Presidio/Marina/Pacific Heights': 'Presidio/Marina/Pacific Heights',
            'Hayes/Nopa/Haight': 'Hayes/Nopa/Haight',
            'Hayes/Napa/Haight': 'Hayes/Nopa/Haight',
            'Mid Market/SoMa': 'Mid Market/SoMa',
            'Mid Market/SOMA': 'Mid Market/SoMa',
            'Mission Bay/Potrero': 'Mission Bay/Potrero',
            'Noe/Mission/Castro': 'Noe/Mission/Castro',
            'Russian Hill/Nob Hill/North Beach': 'Russian Hill/Nob Hill/North Beach',
            'Richmond/Sunset': 'Richmond/Sunset',
            'Excelsior/Glen Park/Bernal': 'Excelsior/Glen Park/Bernal',
            'Buena Vista Park / Ashbury Heights': 'Buena Vista Park/Ashbury Heights',
            'Buena Vista Park/Ashbury Heights': 'Buena Vista Park/Ashbury Heights'
        }
    
    # Create fallback region code mapping if not loaded from file
    if not region_code_mapping:
        region_code_mapping = {
            'South Florida': 'SFL',
            'Boston': 'BOS',
            'New York': 'NYC',
            'Washington DC': 'WDC',
            'Atlanta': 'ATL',
            'Chicago': 'CHI',
            'Houston': 'HOU',
            'Austin': 'AUS',
            'San Francisco': 'SFO',
            'East Bay': 'EBA',
            'Peninsula': 'PSA',
            'Marin County': 'MAR',
            'Napa-Sonoma': 'NAP',
            'Los Angeles': 'LAX',
            'San Diego': 'SAN',
            'Seattle': 'SEA',
            'London': 'LON',
            'Sao Paulo': 'SPO',
            'Mexico City': 'MEX',
            'Singapore': 'SIN',
            'Shanghai': 'SHA',
            'Nairobi': 'NBO',
            'Virtual-Only Americas': 'AM',
            'Virtual-Only APAC+EMEA': 'AE'
        }
    
    return region_mapping, subregion_mapping, region_code_mapping, region_subregion_mapping

# Load the mappings
REGION_MAPPING, SUBREGION_MAPPING, REGION_CODE_MAPPING, REGION_SUBREGION_MAPPING = load_normalization_tables()

def get_region_code_with_subregion(region, subregion, is_virtual=False):
    """
    Get the region code based on region and subregion, with special handling for virtual circles.
    
    Args:
        region: Normalized region name
        subregion: Normalized subregion name
        is_virtual: Whether this is a virtual circle
        
    Returns:
        Region code string
    """
    # For virtual circles, the region code depends on the subregion
    if is_virtual:
        # Try the combined mapping first
        key = (region, subregion)
        if key in REGION_SUBREGION_MAPPING:
            return REGION_SUBREGION_MAPPING[key]
        
        # If not found in combined mapping, check if region starts with 'Virtual'
        if isinstance(region, str) and region.startswith('Virtual'):
            # Check if this is Americas or APAC+EMEA
            if 'Americas' in region:
                # Try to extract timezone from subregion
                if isinstance(subregion, str):
                    if 'GMT-3' in subregion:
                        return 'AM-GMT-3'
                    elif 'GMT-4' in subregion:
                        return 'AM-GMT-4'
                    elif 'GMT-5' in subregion:
                        return 'AM-GMT-5'
                    elif 'GMT-6' in subregion:
                        return 'AM-GMT-6'
                    elif 'GMT-7' in subregion:
                        return 'AM-GMT-7'
                    elif 'GMT-8' in subregion:
                        return 'AM-GMT-8'
                # Default for Americas
                return 'AM'
            elif 'APAC' in region or 'EMEA' in region:
                # Try to extract timezone from subregion
                if isinstance(subregion, str):
                    if 'GMT+1' in subregion:
                        return 'AE-GMT+1'
                    elif 'GMT+2' in subregion:
                        return 'AE-GMT+2'
                    elif 'GMT+3' in subregion:
                        return 'AE-GMT+3'
                    elif 'GMT+4' in subregion:
                        return 'AE-GMT+4'
                    elif 'GMT+5' in subregion or 'GMT+5:30' in subregion:
                        return 'AE-GMT+530'
                    elif 'GMT+7' in subregion:
                        return 'AE-GMT+7'
                    elif 'GMT+8' in subregion:
                        return 'AE-GMT+8'
                    elif 'GMT+9' in subregion:
                        return 'AE-GMT+9'
                    elif 'GMT+10' in subregion:
                        return 'AE-GMT+10'
                    elif 'GMT+12' in subregion:
                        return 'AE-GMT+12'
                    elif 'GMT' in subregion:
                        return 'AE-GMT'
                # Default for APAC+EMEA
                return 'AE'
        
        # If all else fails, use the legacy mapping
        if region in REGION_CODE_MAPPING:
            return REGION_CODE_MAPPING[region]
        
        return 'Invalid'
    
    # For in-person circles, use standard region code from the mapping
    if region in REGION_CODE_MAPPING:
        return REGION_CODE_MAPPING[region]
    
    # If no mapping found, use a default or placeholder
    print(f"⚠️ WARNING: Could not find region code for {region} (subregion: {subregion})")
    return 'Unknown'

def normalize_regions(region):
    """
    Normalize a region name using the mapping table
    
    Args:
        region: Region name to normalize
        
    Returns:
        Normalized region name
    """
    # Handle null values and empty strings
    if pd.isna(region) or not region:
        return ''
    
    # Ensure we have a string
    try:
        region = str(region).strip()
    except Exception as e:
        print(f"Error converting region to string: {e}")
        return str(region) if region is not None else ''
    
    # Direct mapping lookup
    if region in REGION_MAPPING:
        return REGION_MAPPING[region]
    
    # Try case-insensitive matching
    for key, value in REGION_MAPPING.items():
        if isinstance(key, str) and region.lower() == key.lower():
            return value
    
    # If no match, apply basic normalization and return
    normalized = region.strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Replace multiple spaces with single space
    
    return normalized

def normalize_subregions(subregion):
    """
    Normalize a subregion name using the mapping table
    
    Args:
        subregion: Subregion name to normalize
        
    Returns:
        Normalized subregion name
    """
    # Handle null values and empty strings
    if pd.isna(subregion) or not subregion:
        return ''
    
    # Ensure we have a string
    try:
        subregion = str(subregion).strip()
    except Exception as e:
        print(f"Error converting subregion to string: {e}")
        return str(subregion) if subregion is not None else ''
    
    # Direct mapping lookup
    if subregion in SUBREGION_MAPPING:
        return SUBREGION_MAPPING[subregion]
    
    # Try case-insensitive matching
    for key, value in SUBREGION_MAPPING.items():
        if isinstance(key, str) and subregion.lower() == key.lower():
            return value
    
    # If no match, apply basic normalization and return
    normalized = subregion.strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Replace multiple spaces with single space
    normalized = re.sub(r'\s*\/\s*', '/', normalized)  # Standardize slash formatting
    normalized = re.sub(r'\s*-\s*', '-', normalized)  # Standardize hyphen formatting
    
    return normalized

def get_region_code(region):
    """
    Get the region code for a given region name
    
    Args:
        region: Region name (can be normalized or unnormalized)
        
    Returns:
        Region code (e.g., 'LAX' for 'Los Angeles')
    """
    # First normalize the region name
    normalized_region = normalize_regions(region)
    
    # Look up the code in the mapping
    if normalized_region in REGION_CODE_MAPPING:
        return REGION_CODE_MAPPING[normalized_region]
    
    # If not found, generate a fallback code (first 3 letters)
    if normalized_region:
        # Remove spaces and special characters
        clean_region = re.sub(r'[^a-zA-Z0-9]', '', normalized_region)
        # Take first 3 characters and uppercase
        fallback_code = clean_region[:3].upper()
        return fallback_code
    
    # Default fallback
    return 'UNK'  # Unknown
