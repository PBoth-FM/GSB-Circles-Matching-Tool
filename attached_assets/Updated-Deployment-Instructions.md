
# GSB CirclesTool: Updated Deployment Instructions
GSB Circles Matching Tool
Intranet deployment instructions - Updated Version

## About the Tool
The Circles Matching Tool automates the first-round matching for GSB Women's circles. It uses a mathematical algorithm that optimizes for matching as many participants to circles as possible based on location and day/time compatibility, taking into account minimum and maximum circle sizes and hosting requirements. Within those parameters, it also looks for matches that allow for better circle diversity across specified demographic and employment categories. The tool is built in Python and uses the PuLP library for mathematical optimization with a Streamlit web interface.

## Updated Intranet Deployment Instructions

### System Requirements
- Python 3.11 or later
- 8GB or less total project size (Replit deployment limitation)
- Network access for package installation

### Step 1: Install Python and Dependencies

On the server/machine where you want to host the application:

**a. Install Python 3.11 or later**

**b. Install the required packages:**
```bash
pip install numpy>=2.2.4 pandas>=2.2.3 plotly>=6.0.1 pulp>=3.1.1 streamlit>=1.44.1 openai>=1.76.0
```

### Step 2: Copy All Required Files

Copy all the code files maintaining the same directory structure. **All files are now considered required** for proper functionality:

#### Core Application Files
```
app.py
documentation.md
debug_null_id.py
```

#### Module Directory (modules/)
```
modules/__init__.py
modules/circle_reconstruction.py
modules/co_leader_assignment.py
modules/data_loader.py
modules/data_processor.py
modules/demographic_processor.py
modules/diagnostic_tools.py
modules/optimizer.py
modules/optimizer_fixes.py
modules/optimizer_new.py
modules/same_person_constraint_test.py
modules/ui_components.py
modules/ui_components_add.py
```

#### Utilities Directory (utils/)
```
utils/__init__.py
utils/circle_id_postprocessor.py
utils/circle_metadata_manager.py
utils/data_standardization.py
utils/debug_snapshot.py
utils/feature_flags.py
utils/helpers.py
utils/metadata_manager.py
utils/normalization.py
utils/region_mapper.py
utils/validators.py
```

#### Configuration Files
```
.streamlit/config.toml
pyproject.toml
```

#### Data Assets Directory (attached_assets/)
**Updated file names:**
```
attached_assets/Circles-RegionNormalization.csv
attached_assets/Circles-SubregionNormalization.csv
attached_assets/Circles-RegionSubregionCodeMapping.csv
```

#### Debug Data Directory (debug_data/)
```
debug_data/circle_eligibility_logs.json
debug_data/circle_eligibility_logs.json.bak
debug_data/continuing_matching_outcomes.json
debug_data/continuing_members_debug.json
```

### Step 3: Run the Application

**Updated command with proper network binding:**
```bash
streamlit run app.py --server.port 5000 --server.address 0.0.0.0
```

The application will be accessible at:
- Within the intranet: `http://<server-ip>:5000`
- On the host machine: `http://localhost:5000`

### Important Configuration Notes

**Network Configuration:**
- The server IP address must be accessible to users within the intranet
- Port 5000 must be allowed through any firewalls
- The `--server.address 0.0.0.0` flag is required for intranet access

**Streamlit Configuration (.streamlit/config.toml):**
The application includes configuration for:
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

## Updated Dependencies and Verification

### Verified Current Dependencies
Based on the current `pyproject.toml` and `uv.lock` files:

**Core Dependencies:**
- `numpy>=2.2.4` - Numerical computing
- `pandas>=2.2.3` - Data manipulation and analysis
- `plotly>=6.0.1` - Interactive data visualization
- `pulp>=3.1.1` - Linear programming optimization
- `streamlit>=1.44.1` - Web application framework
- `openai>=1.76.0` - AI integration capabilities

**Additional Dependencies (automatically installed):**
- `altair>=5.5.0` - Statistical visualization (Streamlit dependency)
- `pillow>=11.1.0` - Image processing
- `pyarrow>=19.0.1` - Data serialization
- `protobuf>=5.29.4` - Data serialization
- `tenacity>=9.1.2` - Retry utilities
- `typing-extensions>=4.13.1` - Type hints
- `requests>=2.32.3` - HTTP requests
- `packaging>=24.2` - Package utilities

### New Features and Enhancements

**Enhanced Optimization Engine:**
- ID-based matching algorithms (`optimizer_new.py`)
- Co-leader assignment logic (`co_leader_assignment.py`)
- Same-person constraint validation (`same_person_constraint_test.py`)

**Improved Data Management:**
- Circle metadata management (`circle_metadata_manager.py`)
- Data standardization utilities (`data_standardization.py`)
- Feature flag system (`feature_flags.py`)

**Enhanced UI Components:**
- Comprehensive demographics analysis
- Debug and diagnostic tools
- Circle eligibility tracking
- Advanced validation reporting

## Troubleshooting

### Common Issues and Solutions

**1. Missing Dependencies**
If you encounter import errors, ensure all packages are installed:
```bash
pip install --upgrade numpy pandas plotly pulp streamlit openai
```

**2. File Access Issues**
Ensure all directories have proper read permissions:
```bash
chmod -R 755 modules/ utils/ attached_assets/ debug_data/ .streamlit/
```

**3. Port Binding Issues**
If port 5000 is unavailable, change the port in the command:
```bash
streamlit run app.py --server.port 8080 --server.address 0.0.0.0
```

**4. Data File Issues**
Verify the updated CSV files are present:
- `Circles-RegionNormalization.csv` (formerly `Appendix2-RegionNormalizationCodes.csv`)
- `Circles-SubregionNormalization.csv` (formerly `Appendix1-SubregionNormalization.csv`)
- `Circles-RegionSubregionCodeMapping.csv` (new file)

## Appendix

### FAQ

**Q: Is it AI?**
A: The application uses deterministic mathematical optimization, not generative AI. It includes optional AI integration capabilities for enhanced analysis but the core matching algorithm is deterministic and reproducible.

**Q: Was there any generative AI involved?**
A: Replit AI Assistant was used in developing parts of the Python code for enhanced functionality and debugging.

**Q: How does it work?**
A: The algorithm systematically compares all viable circle match combinations, selecting scenarios with the highest optimization score. The scoring system assigns:
- 100 points for each successful match
- 30, 20, or 10 points for 1st, 2nd, or 3rd preference location and day/time matches
- 1 point for each additional category of diversity in a circle

The system prioritizes successful matches first, then location/time preferences, and finally seeks to improve circle diversity within those constraints.

**Q: What's new in this version?**
A: Enhanced optimization algorithms, improved data validation, co-leader assignment automation, constraint violation detection, comprehensive metadata management, and expanded debugging capabilities.

### Support and Documentation

For additional help:
- Check the in-app Documentation tab
- Enable Debug Mode for detailed logging
- Review the comprehensive debug data in the Debug tab
- Contact: Patricia Bothwell (GSB '05) - pbothwell@fastmail.com
