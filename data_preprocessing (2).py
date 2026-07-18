import numpy as np
import pandas as pd

# Load your perfectly cleaned dataset
df = pd.read_csv('USD_Economy_Centurion_Real_Data_v2_PROCESSED.csv', index_col='Date', parse_dates=True)

# ==========================================
# 🎯 STEP 1: DEFINE TARGET & CREATE LABELS
# ==========================================
# Example: Predict if S&P 500 (US_Stock_SP500) goes UP in the next 1 day
# We need to recreate the raw target from the scaled data or use log returns if preferred.
# For simplicity here, we assume we want to predict the direction of 'US_Stock_SP500'

TARGET_COL = 'US_Stock_SP500'
PREDICT_HORIZON = 1 # Predict 1 day ahead

# Create the Target: 1 if return > 0, else 0
# Note: Since data is scaled, positive change in scaled data = positive change in raw data
df['Target_Return'] = df[TARGET_COL].diff(PREDICT_HORIZON).shift(-PREDICT_HORIZON)
df['Target_Class'] = (df['Target_Return'] > 0).astype(int)

# Drop the last 'PREDICT_HORIZON' rows as they have no target (NaN)
df_preprocessed = df.dropna()

print(f"Target created. Predicting {PREDICT_HORIZON}-day direction of {TARGET_COL}.")
print(f"Class Balance: {df_preprocessed['Target_Class'].value_counts(normalize=True)}")

# ==========================================
# ✂️ STEP 2: CHRONOLOGICAL TRAIN/TEST SPLIT
# ==========================================
# 80% Train, 10% Validation, 10% Test
train_size = int(len(df_preprocessed) * 0.8)
val_size = int(len(df_preprocessed) * 0.1)

train_data = df_preprocessed.iloc[:train_size]
val_data = df_preprocessed.iloc[train_size:train_size+val_size]
test_data = df_preprocessed.iloc[train_size+val_size:]

print(f"\nData Splits:")
print(f"Train: {train_data.index.min()} to {train_data.index.max()} ({len(train_data)} rows)")
print(f"Val:   {val_data.index.min()} to {val_data.index.max()} ({len(val_data)} rows)")
print(f"Test:  {test_data.index.min()} to {test_data.index.max()} ({len(test_data)} rows)")

# ==========================================
# 📦 STEP 3: CREATE SEQUENCES (For LSTM/DL)
# ==========================================
def create_sequences(data, feature_cols, target_col, time_steps=30):
    """
    Creates 3D sequences (Samples, Time Steps, Features) for LSTM.
    """
    X, y = [], []
    data_array = data[feature_cols].values
    target_array = data[target_col].values
    
    for i in range(len(data) - time_steps):
        X.append(data_array[i:(i + time_steps)])
        y.append(target_array[i + time_steps])
        
    return np.array(X), np.array(y)

# Define features (exclude target columns)
feature_cols = [c for c in df_preprocessed.columns if 'Target' not in c]
TIME_STEPS = 60 # Look back 60 days

# Create sequences for each split
X_train, y_train = create_sequences(train_data, feature_cols, 'Target_Class', TIME_STEPS)
X_val, y_val = create_sequences(val_data, feature_cols, 'Target_Class', TIME_STEPS)
X_test, y_test = create_sequences(test_data, feature_cols, 'Target_Class', TIME_STEPS)

print(f"\n3D Sequence Shapes (for LSTM Input):")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape},   y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

print("\n✅ PREPROCESSING COMPLETE. Ready for Model Training.")