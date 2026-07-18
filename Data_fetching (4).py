import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
LOGIN = 52642053          # Replace with your ICMarkets MT5 login
PASSWORD = "NQn0uf!lepOleu" # Replace with your password
SERVER = "ICMarketsSC-Demo"  # Check if this is "Demo" or "Live"

SYMBOL = "EURUSD"   # You can change symbol here
TIMEFRAMES = {
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1
}

# Updated Date Range: 10 Dec 2019 to 8 Dec 2025
START_DATE = datetime(2019, 12, 10)
END_DATE   = datetime(2025, 12, 8)

SAVE_FOLDER = "data_icm"  # Folder where CSVs will be saved
MERGE_FILE  = "EURUSD_Merged_2019_2025.csv"  # Final merged file name


# -------------------------------
# INIT MT5 CONNECTION
# -------------------------------
if not mt5.initialize(login=LOGIN, password=PASSWORD, server=SERVER):
    print("❌ Initialize() failed, error:", mt5.last_error())
    quit()
else:
    print("✅ Connected to ICMarkets MT5")


# -------------------------------
# CREATE SAVE FOLDER
# -------------------------------
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


# -------------------------------
# DOWNLOAD FUNCTION
# -------------------------------
def download_data(symbol, timeframe, tf_name):
    print(f"⏳ Downloading {symbol} {tf_name} data...")

    # Copy rates from start to end date
    rates = mt5.copy_rates_range(symbol, timeframe, START_DATE, END_DATE)
    
    if rates is None or len(rates) == 0:
        print(f"❌ No data for {symbol} {tf_name}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Save individual file (1 of the requested files)
    file_name = f"{symbol}_{tf_name}_2019_2025.csv"
    file_path = os.path.join(SAVE_FOLDER, file_name)
    df.to_csv(file_path, index=False)

    print(f"✅ Saved {symbol} {tf_name} -> {file_path}")
    return df


# -------------------------------
# DOWNLOAD FOR ALL TIMEFRAMES
# -------------------------------
all_data = []

# Loop through the 3 timeframes (H4, D1, W1)
for tf_name, tf_code in TIMEFRAMES.items():
    df_tf = download_data(SYMBOL, tf_code, tf_name)
    if df_tf is not None:
        # Add a column so we can distinguish timeframes in the merged file
        df_tf['timeframe'] = tf_name
        all_data.append(df_tf)


# -------------------------------
# MERGE ALL INTO ONE FILE
# -------------------------------
if all_data:
    # Concatenate all dataframes
    merged_df = pd.concat(all_data, axis=0)
    
    # Sort by time to keep it organized
    merged_df = merged_df.sort_values(by="time")
    
    # Save the merged file (The 4th file)
    merge_path = os.path.join(SAVE_FOLDER, MERGE_FILE)
    merged_df.to_csv(merge_path, index=False)
    
    print("-" * 30)
    print(f"📂 Final merged dataset saved as: {merge_path}")
    print(f"📊 Total files created: {len(all_data)} separate + 1 merged")
else:
    print("⚠️ No data downloaded, nothing to merge.")

# -------------------------------
# SHUTDOWN MT5 CONNECTION
# -------------------------------
mt5.shutdown()