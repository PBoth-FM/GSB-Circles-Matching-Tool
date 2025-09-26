#!/usr/bin/env python3
"""
Test script to verify that region normalization works correctly with new regions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.normalization import normalize_regions, normalize_subregions, get_region_code_with_subregion, load_normalization_tables, get_region_code

def test_new_regions():
    """Test the new regions: Tel Aviv, Toronto, UAE"""
    print("🧪 Testing new region normalization...")
    
    # Test cases for new regions
    test_cases = [
        # (input_region, expected_normalized, expected_code)
        ("Tel Aviv", "Tel Aviv", "TVL"),
        ("Toronto", "Toronto", "TOR"), 
        ("United Arab Emirates", "United Arab Emirates", "UAE"),
        ("UAE", "United Arab Emirates", "UAE"),
        ("Dubai", "United Arab Emirates", "UAE"),
    ]
    
    for input_region, expected_normalized, expected_code in test_cases:
        normalized = normalize_regions(input_region)
        code = get_region_code(normalized)
        
        print(f"  {input_region} -> {normalized} (code: {code})")
        
        if normalized != expected_normalized:
            print(f"    ❌ FAIL: Expected '{expected_normalized}', got '{normalized}'")
        else:
            print(f"    ✅ PASS: Normalization correct")
            
        if code != expected_code:
            print(f"    ❌ FAIL: Expected code '{expected_code}', got '{code}'")
        else:
            print(f"    ✅ PASS: Region code correct")

def test_virtual_region_split():
    """Test that Virtual EMEA and APAC are now handled separately"""
    print("\n🧪 Testing virtual region split...")
    
    # Test cases for virtual regions
    test_cases = [
        # (region, subregion, expected_code_prefix)
        ("Virtual EMEA", "GMT+1 (Central European Time)", "EM"),
        ("Virtual EMEA", "GMT (Greenwich Mean Time)", "EM"),
        ("Virtual APAC", "GMT+8 (Singapore Standard Time)", "AP"),
        ("Virtual APAC", "GMT+9 (Japan Standard Time)", "AP"),
        ("Virtual-Only Americas", "GMT-5 (Eastern Standard Time)", "AM"),
    ]
    
    for region, subregion, expected_prefix in test_cases:
        code = get_region_code_with_subregion(region, subregion, is_virtual=True)
        
        print(f"  {region} / {subregion} -> {code}")
        
        if code.startswith(expected_prefix):
            print(f"    ✅ PASS: Correct prefix {expected_prefix}")
        else:
            print(f"    ❌ FAIL: Expected prefix '{expected_prefix}', got '{code}'")

def test_csv_loading():
    """Test that CSV normalization files are loading correctly"""
    print("\n🧪 Testing CSV file loading...")
    
    try:
        region_mapping, subregion_mapping, region_code_mapping, region_subregion_mapping = load_normalization_tables()
        
        print(f"  Region mappings loaded: {len(region_mapping)} entries")
        print(f"  Subregion mappings loaded: {len(subregion_mapping)} entries") 
        print(f"  Region code mappings loaded: {len(region_code_mapping)} entries")
        print(f"  Region-subregion mappings loaded: {len(region_subregion_mapping)} entries")
        
        # Check for new regions in mappings
        new_regions = ["Tel Aviv", "Toronto", "United Arab Emirates"]
        new_codes = ["TVL", "TOR", "UAE", "EM", "AP"]
        
        for region in new_regions:
            if region in region_mapping:
                print(f"    ✅ PASS: {region} found in region mapping")
            else:
                print(f"    ⚠️  INFO: {region} not in region mapping (may use fallback)")
                
        for code in new_codes:
            found_in_mapping = any(code in str(v) for v in region_code_mapping.values())
            if found_in_mapping or code in region_code_mapping.values():
                print(f"    ✅ PASS: Code {code} found in mappings")
            else:
                print(f"    ⚠️  INFO: Code {code} not in mappings (may use fallback)")
                
    except Exception as e:
        print(f"    ❌ FAIL: Error loading CSV files: {e}")

if __name__ == "__main__":
    print("=== Region Update Verification Test ===\n")
    
    test_new_regions()
    test_virtual_region_split() 
    test_csv_loading()
    
    print("\n✅ Testing complete!")