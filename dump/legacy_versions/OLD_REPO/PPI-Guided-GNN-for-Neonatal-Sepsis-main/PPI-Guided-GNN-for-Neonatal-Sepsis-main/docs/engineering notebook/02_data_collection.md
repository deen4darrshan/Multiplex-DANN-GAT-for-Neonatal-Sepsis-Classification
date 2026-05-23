# Data Collection

Collection rules
- Only public datasets were used.
- Each dataset is downloaded from the GEO FTP location if not already cached.
- Raw files are preserved in `data\raw` and never modified in place.

Download approach
- `scripts\00_download_data.py` fetches GSE160207 and GSE163812 raw counts and series matrix files.
- `scripts\07_prepare_multicohort_real.py` fetches the multi cohort files for GSE160207, GSE163812, GSE180838, and GSE186141.
- `scripts\09_prepare_expanded_real.py` fetches the mouse dataset for GSE154748.

Metadata extraction
- Series matrix files are parsed to recover labels, sample titles, and characteristics.
- For GSE163812, treatment labels are parsed and only GFP baseline samples are retained.
- For GSE160207, family identifiers are extracted and used as GroupID for grouped cross validation.

Reasoning
- Keeping raw files intact allows reproducibility of upstream steps.
- The metadata parsing is explicitly recorded to avoid hidden label leakage.

Key files
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\00_download_data.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\07_prepare_multicohort_real.py`
- `C:\Users\terry\Downloads\Projects\ISEF\Osteogenesis imperfecta\scripts\09_prepare_expanded_real.py`
