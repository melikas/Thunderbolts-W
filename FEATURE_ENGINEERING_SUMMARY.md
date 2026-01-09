# BLE Data Feature Engineering - 10-Second Time Windows

## Summary

Successfully transformed the raw BLE dataset into a time-windowed feature table using 10-second non-overlapping windows.

## Output File

- **Filename**: `ble_data_FE_labeled.csv`
- **Location**: `c:\Users\umroot\Desktop\BLE Data\`
- **File Size**: 1,639.61 KB
- **Total Rows**: 4,969 (10-second windows)
- **Total Columns**: 77

## Data Structure

### Column Breakdown

1. **Column 1: Time** (1 column)
   - `time`: Window start timestamp (format: YYYY-MM-DD HH:MM:SS)
   - Time range: 2023-04-10 10:22:50 to 2023-04-13 13:56:20

2. **Columns 2-26: Beacon Count Features** (25 columns)
   - `count_beacon_1` to `count_beacon_25`
   - Each column contains the count of detections for that beacon ID in the 10-second window
   - Range: 0 to 273 detections per beacon per window

3. **Columns 27-76: RSSI Statistics** (50 columns)
   - For each beacon (1-25), two features:
     - `mean_rssi_b1` to `mean_rssi_b25`: Mean RSSI value
     - `sd_rssi_b1` to `sd_rssi_b25`: Standard deviation of RSSI
   - RSSI values range: approximately -60 to -98 dBm
   - SD provides signal stability measurement

4. **Column 77: Location** (1 column)
   - `location`: The room/location label for the 10-second window
   - 39 unique locations identified
   - 3,153 labeled windows (63.5% coverage)
   - 1,816 unlabeled windows marked as "unknown"

## Feature Engineering Details

### Time Windows
- **Window Size**: 10 seconds, non-overlapping
- **Method**: Floor timestamps to nearest 10-second boundary
- **Coverage**: All data points from April 10-13, 2023

### Beacon Processing
- **Total unique beacons**: 4,832 MAC addresses detected
- **Selected beacons**: 25 most frequent beacon IDs
- **Data included**: 1,328,379 data points mapped to 25 beacons

### RSSI Filtering
- Invalid RSSI values (0, extreme values < -100) excluded from statistics
- Mean and SD calculated only from valid RSSI readings
- 0.0 values indicate no beacon detections in that window

## Location Distribution (Top 10)

| Location | Windows | Percentage |
|----------|---------|-----------|
| unknown | 1,816 | 36.5% |
| nurse station | 778 | 15.7% |
| kitchen | 527 | 10.6% |
| Office Small | 360 | 7.2% |
| cafeteria | 358 | 7.2% |
| Cafeteria D | 212 | 4.3% |
| Office Large | 172 | 3.5% |
| hallway | 153 | 3.1% |
| 201 | 80 | 1.6% |
| 213 | 69 | 1.4% |

## Data Quality

✓ **Complete**: No missing values in time or location columns
✓ **Validated**: All 77 columns present and populated
✓ **Consistent**: 10-second window boundaries maintained
✓ **Labeled**: 63.5% of windows have location labels

## Usage Example

The output file can be used for:

1. **Machine Learning**: Train location classification models
2. **Pattern Analysis**: Analyze beacon signal patterns by location
3. **Signal Quality**: Study RSSI stability (via SD) across locations
4. **Temporal Patterns**: Investigate movement patterns over time

## Processing Details

- **Source Files Used**:
  - BLEdata.csv (merged BLE data, 5,005,751 rows)
  - 5f_label_loc_train.csv (location labels, 1,334 rows)

- **Processing Steps**:
  1. Parsed timestamps and created 10-second window identifiers
  2. Mapped 4,832 unique beacon MAC addresses to beacon IDs 1-25
  3. Aggregated beacon counts and RSSI statistics per window
  4. Matched time windows to location labels
  5. Organized columns: time → beacon counts → RSSI stats → location

- **Computation**: Successfully completed with 4,969 output windows
