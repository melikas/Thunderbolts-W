"""
BLE Data Feature Engineering Version 2
Enhanced feature extraction with additional statistical features
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configuration
BASE_DIR = r"C:\Users\laber\MASc\Thunderbolts-W"
DATA_DIR = os.path.join(BASE_DIR, "data")
BLE_DATA_FILE = os.path.join(DATA_DIR, "ble_data_labeled.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "ble_data_FE2_labeled.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Constants
WINDOW_SIZE = 10  # seconds
NUM_BEACONS = 25

print("=" * 80)
print("BLE Data Feature Engineering Version 2 - Enhanced Features")
print("=" * 80)

# Step 1: Load the data
print("\n[Step 1] Loading BLE data...")
ble_df = pd.read_csv(BLE_DATA_FILE)
print(f"  - Loaded {len(ble_df):,} records")
print(f"  - Columns: {ble_df.shape[1]}")

# Step 2: Prepare timestamps and windows
print("\n[Step 2] Creating time windows...")
ble_df['timestamp'] = pd.to_datetime(ble_df['timestamp'])
ble_df = ble_df.sort_values('timestamp').reset_index(drop=True)
ble_df['window_start'] = ble_df['timestamp'].dt.floor('10s')
print(f"  - Created {ble_df['window_start'].nunique()} unique 10-second windows")

# Step 3: Map beacons
print("\n[Step 3] Mapping beacon IDs...")
unique_macs = ble_df['mac_address'].unique()
mac_to_beacon_id = {mac: i+1 for i, mac in enumerate(unique_macs[:NUM_BEACONS])}
ble_df['beacon_id'] = ble_df['mac_address'].map(mac_to_beacon_id)
ble_df_filtered = ble_df[ble_df['beacon_id'].notna()].copy()
print(f"  - Processing {len(ble_df_filtered):,} records with {len(mac_to_beacon_id)} beacons")

# Step 4: Enhanced feature extraction
print("\n[Step 4] Extracting enhanced features...")

feature_data = []

for window_start, group in ble_df_filtered.groupby('window_start'):
    row_data = {'time': window_start}
    
    # Get location (most common room in window)
    if 'room' in group.columns:
        location_counts = group['room'].value_counts()
        row_data['location'] = location_counts.index[0] if len(location_counts) > 0 else 'unknown'
    else:
        row_data['location'] = 'unknown'
    
    # Initialize all features (counts multiplied into statistical features)
    for beacon_id in range(1, NUM_BEACONS + 1):
        # RSSI statistical features (will be multiplied by count)
        row_data[f'mean_rssi_b{beacon_id}'] = 0
        row_data[f'std_rssi_b{beacon_id}'] = 0
        row_data[f'min_rssi_b{beacon_id}'] = 0
        row_data[f'max_rssi_b{beacon_id}'] = 0
        row_data[f'median_rssi_b{beacon_id}'] = 0
    
    # Calculate features for each beacon (count multiplied into statistical features)
    for beacon_id in range(1, NUM_BEACONS + 1):
        beacon_data = group[group['beacon_id'] == beacon_id]
        
        if len(beacon_data) > 0:
            count = len(beacon_data)
            
            # RSSI statistics multiplied by count
            rssi_values = beacon_data['RSSI'].astype(float)
            rssi_valid = rssi_values[(rssi_values < 0) & (rssi_values > -100)]
            
            if len(rssi_valid) > 0:
                row_data[f'mean_rssi_b{beacon_id}'] = rssi_valid.mean() * count
                row_data[f'std_rssi_b{beacon_id}'] = (rssi_valid.std() if len(rssi_valid) > 1 else 0.0) * count
                row_data[f'min_rssi_b{beacon_id}'] = rssi_valid.min() * count
                row_data[f'max_rssi_b{beacon_id}'] = rssi_valid.max() * count
                row_data[f'median_rssi_b{beacon_id}'] = rssi_valid.median() * count
    
    feature_data.append(row_data)

features_df = pd.DataFrame(feature_data)
print(f"  - Created {len(features_df)} feature vectors")
print(f"  - Total features per window: {features_df.shape[1]}")

# Step 5: Organize columns
print("\n[Step 5] Organizing output...")

# Build column order (no separate count columns)
rssi_feature_cols = []
for i in range(1, NUM_BEACONS + 1):
    rssi_feature_cols.extend([
        f'mean_rssi_b{i}',
        f'std_rssi_b{i}',
        f'min_rssi_b{i}',
        f'max_rssi_b{i}',
        f'median_rssi_b{i}'
    ])

output_cols = ['time'] + rssi_feature_cols + ['location']
output_df = features_df[output_cols].copy()

print(f"  - Column structure:")
print(f"    * time: 1 column")
print(f"    * RSSI features (count-weighted): {len(rssi_feature_cols)} columns")
print(f"      (mean, std, min, max, median × count for each beacon)")
print(f"    * location: 1 column")
print(f"    * Total: {len(output_cols)} columns")

# Step 6: Save output
print("\n[Step 6] Saving enhanced features...")
output_df.to_csv(OUTPUT_FILE, index=False)
file_size = os.path.getsize(OUTPUT_FILE) / 1024
print(f"  - Saved to: {OUTPUT_FILE}")
print(f"  - File size: {file_size:.2f} KB")

# Step 7: Summary
print("\n[Step 7] Summary Statistics:")
print(f"  - Total windows: {len(output_df):,}")
print(f"  - Time range: {output_df['time'].min()} to {output_df['time'].max()}")
print(f"  - Unique locations: {output_df['location'].nunique()}")
print(f"  - Labeled windows: {(output_df['location'] != 'unknown').sum():,} ({100*(output_df['location'] != 'unknown').sum()/len(output_df):.1f}%)")

print(f"\n  - Top 5 locations:")
for i, (loc, count) in enumerate(output_df['location'].value_counts().head(5).items(), 1):
    print(f"      {i}. {loc}: {count} ({100*count/len(output_df):.1f}%)")

# Calculate some beacon statistics
beacon_cols = [col for col in output_df.columns if col.startswith('count_beacon_')]
total_detections = output_df[beacon_cols].sum().sum()
avg_active = (output_df[beacon_cols] > 0).sum(axis=1).mean()

print(f"\n  - Beacon statistics:")
print(f"      Total detections: {total_detections:,.0f}")
print(f"      Avg active beacons per window: {avg_active:.1f}")

print("\n" + "=" * 80)
print("✓ Enhanced Feature Engineering Complete!")
print("=" * 80)

# Display sample
print("\nSample output (first 2 rows, selected columns):")
sample_cols = ['time', 'mean_rssi_b1', 'std_rssi_b1', 
               'min_rssi_b1', 'max_rssi_b1', 'median_rssi_b1', 'location']
print(output_df[sample_cols].head(2).to_string(index=False))
print()
