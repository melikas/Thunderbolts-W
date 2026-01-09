"""
BLE Data Feature Engineering with 10-second Time Windows
Transforms BLE data into time-windowed features with beacon counts and RSSI statistics
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration
DATA_DIR = r"c:\Users\umroot\Desktop\BLE Data"
BLE_DATA_FILE = os.path.join(DATA_DIR, "ble_data_labeled.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "ble_data_FE_labeled.csv")

# Constants
WINDOW_SIZE = 10  # seconds
NUM_BEACONS = 25

print("=" * 80)
print("BLE Data Feature Engineering with 10-Second Time Windows")
print("=" * 80)

# Step 1: Load the data
print("\n[Step 1] Loading data...")
print(f"  - Loading BLE data with room labels from: {BLE_DATA_FILE}")
ble_df = pd.read_csv(BLE_DATA_FILE)

print(f"  - BLE data shape: {ble_df.shape}")
print(f"  - Checking for room column: {'room' in ble_df.columns}")

# Step 2: Parse timestamp and create window identifier
print("\n[Step 2] Preparing data...")
ble_df['timestamp'] = pd.to_datetime(ble_df['timestamp'])
ble_df = ble_df.sort_values('timestamp').reset_index(drop=True)

print(f"  - Timestamp range: {ble_df['timestamp'].min()} to {ble_df['timestamp'].max()}")

# Create 10-second window identifier
ble_df['window_start'] = ble_df['timestamp'].dt.floor(f'{WINDOW_SIZE}S')
print(f"  - Created {ble_df['window_start'].nunique()} unique 10-second windows")

# Step 3: Map beacon IDs to beacon numbers (1-25)
print("\n[Step 3] Processing beacon data...")
# Get unique MAC addresses and map them to beacon IDs 1-25
unique_macs = ble_df['mac_address'].unique()
print(f"  - Found {len(unique_macs)} unique beacon MAC addresses")

# If there are more than 25, we'll use only the first 25
# If there are fewer than 25, we'll still create 25 columns
mac_to_beacon_id = {mac: i+1 for i, mac in enumerate(unique_macs[:NUM_BEACONS])}
print(f"  - Mapping {len(mac_to_beacon_id)} beacons to beacon IDs 1-25")

# Add beacon ID column
ble_df['beacon_id'] = ble_df['mac_address'].map(mac_to_beacon_id)

# Filter out entries with unknown beacons (not in our 25)
ble_df_filtered = ble_df[ble_df['beacon_id'].notna()].copy()
print(f"  - Keeping {len(ble_df_filtered)} entries with known beacon IDs")

# Step 4: Feature aggregation by 10-second window
print("\n[Step 4] Aggregating features by 10-second windows...")

feature_data = []

for window_start, group in ble_df_filtered.groupby('window_start'):
    row_data = {'time': window_start}
    
    # Initialize all beacon counts and RSSI stats to 0
    for beacon_id in range(1, NUM_BEACONS + 1):
        row_data[f'count_beacon_{beacon_id}'] = 0
        row_data[f'mean_rssi_b{beacon_id}'] = 0
        row_data[f'sd_rssi_b{beacon_id}'] = 0
    
    # Get the most common location in this window
    if 'room' in group.columns:
        location_counts = group['room'].value_counts()
        if len(location_counts) > 0:
            row_data['location'] = location_counts.index[0]
        else:
            row_data['location'] = 'unknown'
    else:
        row_data['location'] = 'unknown'
    
    # Process each beacon in this window
    for beacon_id in range(1, NUM_BEACONS + 1):
        beacon_data = group[group['beacon_id'] == beacon_id]
        
        if len(beacon_data) > 0:
            # Count occurrences
            row_data[f'count_beacon_{beacon_id}'] = len(beacon_data)
            
            # Calculate RSSI statistics
            rssi_values = beacon_data['RSSI'].astype(float)
            # Filter out invalid RSSI values (like 0 or very extreme values)
            rssi_valid = rssi_values[(rssi_values < 0) & (rssi_values > -100)]
            
            if len(rssi_valid) > 0:
                row_data[f'mean_rssi_b{beacon_id}'] = rssi_valid.mean()
                row_data[f'sd_rssi_b{beacon_id}'] = rssi_valid.std() if len(rssi_valid) > 1 else 0.0
            else:
                row_data[f'mean_rssi_b{beacon_id}'] = 0
                row_data[f'sd_rssi_b{beacon_id}'] = 0
    
    feature_data.append(row_data)

features_df = pd.DataFrame(feature_data)
print(f"  - Created feature table with {len(features_df)} windows")
print(f"  - Columns: {features_df.shape[1]}")

# Step 5: Verify location labels
print("\n[Step 5] Verifying location labels...")

labeled_count = (features_df['location'] != 'unknown').sum()
print(f"  - Labeled {labeled_count} out of {len(features_df)} windows ({100*labeled_count/len(features_df):.1f}%)")

# Step 6: Reorder columns
print("\n[Step 6] Organizing output columns...")

# Column order: time, beacon counts (25), RSSI stats (50), location
beacon_count_cols = [f'count_beacon_{i}' for i in range(1, NUM_BEACONS + 1)]
rssi_cols = []
for i in range(1, NUM_BEACONS + 1):
    rssi_cols.append(f'mean_rssi_b{i}')
    rssi_cols.append(f'sd_rssi_b{i}')

output_cols = ['time'] + beacon_count_cols + rssi_cols + ['location']
output_df = features_df[output_cols].copy()

print(f"  - Output column order:")
print(f"    1. time (1 column)")
print(f"    2. Beacon counts: count_beacon_1 to count_beacon_25 (25 columns)")
print(f"    3. RSSI statistics: mean_rssi_b1, sd_rssi_b1, ..., mean_rssi_b25, sd_rssi_b25 (50 columns)")
print(f"    4. location (1 column)")
print(f"    - Total: {len(output_cols)} columns")

# Step 7: Save output
print("\n[Step 7] Saving output...")
output_df.to_csv(OUTPUT_FILE, index=False)
print(f"  - Saved to: {OUTPUT_FILE}")
print(f"  - File size: {os.path.getsize(OUTPUT_FILE) / 1024:.2f} KB")

# Step 8: Summary statistics
print("\n[Step 8] Summary Statistics:")
print(f"  - Total windows: {len(output_df)}")
print(f"  - Time range: {output_df['time'].min()} to {output_df['time'].max()}")
print(f"  - Unique locations: {output_df['location'].nunique()}")
print(f"  - Location distribution:")
for loc in output_df['location'].value_counts().head(10).index:
    count = (output_df['location'] == loc).sum()
    pct = 100 * count / len(output_df)
    print(f"      {loc}: {count} ({pct:.1f}%)")

print("\n" + "=" * 80)
print("Feature Engineering Complete!")
print("=" * 80)

# Show sample of output
print("\nSample of output data (first 3 rows):")
print(output_df.head(3).to_string())
