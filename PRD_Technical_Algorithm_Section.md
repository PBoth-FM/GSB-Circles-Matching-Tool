# GSB Circles Matching Tool - Technical Algorithm Documentation

## Optimization Algorithm Implementation

### Core Technology
- **Linear Programming Library**: PuLP (Python)
- **Problem Type**: Binary Integer Linear Programming (BILP)
- **Decision Variables**: x[participant_id, circle_id] ∈ {0,1}
- **Activation Variables**: y[circle_id] ∈ {0,1}

### Complete Objective Function

```python
total_obj = match_obj + very_small_circle_bonus + small_circle_bonus + 
           existing_circle_bonus + pref_obj - new_circle_penalty + 
           special_test_bonus + diversity_bonus
```

#### Component Breakdown

**1. Base Assignment Score (match_obj)**
```python
match_obj = 1000 * pulp.lpSum(x[(p_id, c_id)] for p_id in participants 
                             for c_id in all_circle_ids if (p_id, c_id) in x)
```
- 1000 points per participant successfully matched
- Ensures maximum participant assignment is prioritized

**2. Circle Size Incentives**
```python
# Very small circles (2-3 members) - urgent filling needed
very_small_circle_bonus = 800 * pulp.lpSum(x[(p_id, c_id)] for p_id in participants 
                                          for c_id in very_small_circles_ids if (p_id, c_id) in x)

# Small circles (exactly 4 members) - moderate filling incentive  
small_circle_bonus = 50 * pulp.lpSum(x[(p_id, c_id)] for p_id in participants 
                                    for c_id in small_circles_ids_4 if (p_id, c_id) in x)

# Existing circle preservation bonus
existing_circle_bonus = 500 * pulp.lpSum(x[(p_id, c_id)] for p_id in participants 
                                        for c_id in existing_circle_ids if (p_id, c_id) in x)
```

**3. Preference Scoring (pref_obj)**

*Location Preference Implementation*:
```python
# Exact string matching against participant's ranked choices
if p_row['first_choice_location'] == subregion:
    loc_score = 30
elif p_row['second_choice_location'] == subregion:
    loc_score = 20
elif p_row['third_choice_location'] == subregion:
    loc_score = 10
else:
    loc_score = 0
```

*Time Preference Implementation*:
```python
# Sophisticated time compatibility using is_time_compatible()
if is_time_compatible(first_choice, time_slot, is_important=is_test_case):
    time_score = 30
elif is_time_compatible(second_choice, time_slot, is_important=is_test_case):
    time_score = 20
elif is_time_compatible(third_choice, time_slot, is_important=is_test_case):
    time_score = 10
else:
    time_score = 0
```

**4. Diversity Optimization (diversity_bonus)**
```python
# Weighted by circle activation to only count when circles are used
diversity_bonus += circle_diversity_score * y[c_id]
```

Diversity calculation across five demographic categories:
- **Class_Vintage**: GSB graduation year groupings
- **Employment_Category**: Professional background categories  
- **Industry_Category**: Industry sector groupings
- **Racial_Identity_Category**: Self-identified racial/ethnic categories
- **Children_Category**: Parental status categories

### Constraint Implementation

**Hard Constraints**:
```python
# Each participant assigned to at most one circle
for p_id in participants:
    prob += pulp.lpSum(x[(p_id, c_id)] for c_id in all_circle_ids if (p_id, c_id) in x) <= 1

# Circle capacity limits
for c_id in all_circle_ids:
    max_capacity = circle_metadata[c_id]['max_additions'] + circle_metadata[c_id]['current_members']
    prob += pulp.lpSum(x[(p_id, c_id)] for p_id in participants if (p_id, c_id) in x) <= max_capacity

# Compatibility constraints - only allow valid assignments
for (p_id, c_id) in incompatible_pairs:
    prob += x[(p_id, c_id)] == 0
```

### CURRENT-CONTINUING Member Preprocessing

**Special Pre-assignment Logic**:
```python
# Force CURRENT-CONTINUING members to their existing circles before optimization
if status in ['CURRENT-CONTINUING', 'Current-CONTINUING']:
    current_circle = find_current_circle_id(p_row)
    if current_circle and current_circle in all_circle_ids:
        # Force compatibility with ONLY their current circle
        for c_id in all_circle_ids:
            is_compatible = (c_id == current_circle)
            compatibility[(p_id, c_id)] = 1 if is_compatible else 0
```

### Time Compatibility Logic

**Advanced Time Matching** (`is_time_compatible()`):
- **Exact Matches**: Direct string comparison
- **Day Range Overlaps**: "Monday-Wednesday" matches "Tuesday-Thursday"
- **"Varies" Handling**: "Varies" preference matches any offered time
- **Abbreviation Support**: Mon, Tue, Wed, etc.
- **Case-insensitive**: Handles different capitalizations
- **Special Cases**: Continuing member times vs. official circle times

### Circle Management

**Circle Types**:
- **Existing Circles**: Current circles with CURRENT-CONTINUING members
- **Viable Circles**: Existing circles with `max_additions > 0`
- **Small Circles**: 2-4 member circles needing growth
- **New Circles**: Generated from unmatched participant preferences

**Circle ID Format**: `{FORMAT}-{REGION_CODE}-{IDENTIFIER}`
- FORMAT: "IP" (In-Person) or "VO" (Virtual)
- REGION_CODE: Geographic identifier or timezone for virtual
- IDENTIFIER: Sequential number or "NEW-XX" for newly created

### Optimization Modes

**Existing Circle Handling**:
1. **Preserve Mode**: Existing circles remain intact, new participants only if co-leaders allow
2. **Optimize Mode**: New participants can join existing circles based on algorithm optimization  
3. **Dissolve Mode**: Ignores existing circles, creates all new optimal groupings

This technical implementation ensures the matching algorithm balances participant preferences, circle viability, demographic diversity, and business constraints through sophisticated linear programming optimization.