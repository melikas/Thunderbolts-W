import pandas as pd
import os
import numpy as np

# ============================================================================
# COMPREHENSIVE BLE DATA MERGING 
# ============================================================================
# This script performs the following operations:
# 1. Merges all individual BLE CSV files
# 2. Transforms data structure and adds date/time columns
# 3. Adds RSSI_1..RSSI_25 binary columns per row (no pivot)
# 4. Preserves fractional seconds and ordering; no row drops
# 5. Optionally removes timezone indicator without altering milliseconds
# ============================================================================

def setup_beacon_dictionary():
    """Define MAC address to RSSI column mapping"""
    mac_to_rssi_column = {
        'F7:7F:78:76:7E:F3': 'RSSI_1',
        'C6:CD:5E:3D:2F:BB': 'RSSI_2',
        'D6:F4:3A:79:74:63': 'RSSI_3',
        'C9:17:55:E2:3E:0E': 'RSSI_4',
        'CA:60:AB:EE:EC:7F': 'RSSI_5',
        'D6:51:7F:AB:0E:29': 'RSSI_6',
        'CC:54:33:F6:A7:90': 'RSSI_7',
        'EB:20:56:87:04:5A': 'RSSI_8',
        'EE:E7:46:DC:19:6F': 'RSSI_9',
        'C8:5B:BF:37:07:A0': 'RSSI_10',
        'D7:26:F6:A3:44:D2': 'RSSI_11',
        'DD:83:B0:27:FD:36': 'RSSI_12',
        'E5:CD:4A:36:87:06': 'RSSI_13',
        'DC:22:B8:17:4E:B5': 'RSSI_14',
        'EA:09:20:80:D6:44': 'RSSI_15',
        'E6:99:D1:EC:C6:81': 'RSSI_16',
        'F6:DA:97:C7:D5:28': 'RSSI_17',
        'EA:66:A1:12:2C:F4': 'RSSI_18',
        'C9:EA:57:8B:0F:80': 'RSSI_19',
        'D6:7C:1D:2C:2A:0A': 'RSSI_20',
        'DA:E1:70:5F:44:97': 'RSSI_21',
        'DD:10:10:F6:4F:27': 'RSSI_22',
        'E6:F3:93:A8:9E:22': 'RSSI_23',
        'E6:60:05:1F:88:F9': 'RSSI_24',
        'D4:33:FD:F4:C2:A8': 'RSSI_25'
    }
    return mac_to_rssi_column


def merge_individual_csv_files(dataset_directory):
    """
    STEP 1: Merge all individual BLE CSV files from the dataset directory
    """
    print("=" * 80)
    print("STEP 1: MERGING INDIVIDUAL CSV FILES")
    print("=" * 80)
    
    # Always merge from source CSVs to avoid any prior transformations
    all_files = os.listdir(dataset_directory)
    # Only include raw files that start with the expected prefix
    csv_files = [file for file in all_files if file.endswith('.csv') and file.startswith('user-ble-id_')]
    
    print(f"Found {len(csv_files)} CSV files to merge")
    
    all_dfs = []
    for csv_file in csv_files:
        file_path = os.path.join(dataset_directory, csv_file)
        try:
            df_temp = pd.read_csv(file_path, header=None, dtype={0: 'int64', 1: 'string', 2: 'string', 3: 'string', 4: 'float64', 5: 'string'})
            df_temp.columns = ['pid', 'timestamp', 'column3', 'mac_address', 'rssi', 'column6']
            all_dfs.append(df_temp)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Successfully merged {len(all_dfs)} files")
        print(f"Total rows: {len(df)}\n")
    else:
        print("No CSV files found to merge!")
        return None
    
    return df


def transform_and_flag_data(df, mac_to_rssi_column):
    """
    STEP 2: Transform data structure and add per-row RSSI flags without pivoting
    Preserves every input row and fractional seconds, keeps ordering.
    """
    print("=" * 80)
    print("STEP 2: TRANSFORMING DATA (NO PIVOT)")
    print("=" * 80)
    
    # Convert timestamp to datetime preserving fractional seconds and mixed formats
    print("Converting timestamps...")
    dt = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
    # If timestamps are timezone-aware, strip tz without changing the time
    try:
        dt = dt.dt.tz_localize(None)
    except Exception:
        pass
    df['timestamp'] = dt
    
    # Extract date and time columns
    print("Extracting date and time information...")
    df['year_month_day'] = df['timestamp'].dt.strftime('%Y-%m-%d')
    df['hour'] = df['timestamp'].dt.hour
    
    # Rename columns
    df = df.rename(columns={
        'pid': 'user_id',
        'rssi': 'RSSI',
        'column6': 'power'
    })
    
    # Build RSSI_1..RSSI_25 binary flags per row (1 if this row belongs to that beacon)
    print("Building RSSI_1..RSSI_25 binary columns per row...")
    rssi_cols = [f'RSSI_{i}' for i in range(1, 26)]
    for col in rssi_cols:
        df[col] = 0
    # Map each mac to its target RSSI_* column
    df['rssi_target_col'] = df['mac_address'].map(mac_to_rssi_column)
    # Set flag 1 for the corresponding column for each row
    for col in rssi_cols:
        df.loc[df['rssi_target_col'] == col, col] = 1
    df.drop(columns=['rssi_target_col'], inplace=True)
    
    # Sort to ensure stable ordering: by user_id then timestamp, preserving duplicates
    print("Sorting data by user_id and timestamp while preserving all rows...")
    df = df.sort_values(by=['user_id', 'timestamp'], kind='mergesort').reset_index(drop=True)
    
    # Final column order
    columns_order = ['user_id', 'timestamp', 'mac_address', 'RSSI', 'power', 'year_month_day', 'hour'] + rssi_cols
    df = df[columns_order]
    
    print(f"Data transformed successfully (no pivot). Total rows: {len(df)}\n")
    return df


def convert_rssi_to_binary(df):
    """
    STEP 3: (No-op) RSSI_1..RSSI_25 are already binary flags per row.
    Keeping this for pipeline compatibility.
    """
    print("=" * 80)
    print("STEP 3: RSSI FLAGS ALREADY BINARY – SKIPPING")
    print("=" * 80)
    return df


def save_output(df, output_path):
    """
    STEP 4: Save the final result to CSV file
    """
    print("=" * 80)
    print("STEP 4: SAVING OUTPUT FILE")
    print("=" * 80)
    
    df.to_csv(output_path, index=False)
    
    print(f"File saved successfully: {output_path}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nColumns: {list(df.columns)}\n")
    
    return df


def display_summary(df):
    """Display summary of the processed data"""
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    
    print(f"\nData shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    
    rssi_cols = [f'RSSI_{i}' for i in range(1, 26)]
    print(f"\nRSSI columns sample (binary values):")
    print(df[['user_id', 'timestamp'] + rssi_cols[:10]].head(5))


def main():
    """Main execution function"""
    print("\n")
    print("#" * 80)
    print("# BLE DATA MERGING AND TRANSFORMATION")
    print("#" * 80)
    print("\n")
    
    # Configuration
    dataset_directory = r"c:\Users\umroot\Desktop\BLE Data"
    output_file = r"c:\Users\umroot\Desktop\BLE Data\BLEdata3.csv"
    
    # Get beacon dictionary
    mac_to_rssi_column = setup_beacon_dictionary()
    
    # Step 1: Merge individual CSV files
    df = merge_individual_csv_files(dataset_directory)
    if df is None:
        print("Error: Could not load data")
        return
    
    # Step 2: Transform without pivot; keep all rows
    df = transform_and_flag_data(df, mac_to_rssi_column)
    
    # Step 3: Convert RSSI to binary
    df = convert_rssi_to_binary(df)
    
    # Step 4: Save output
    df = save_output(df, output_file)
    
    # Display summary
    display_summary(df)
    
    print("\n")
    print("#" * 80)
    print("# PROCESS COMPLETED SUCCESSFULLY!")
    print("#" * 80)
    print("\n")


if __name__ == "__main__":
    main()
