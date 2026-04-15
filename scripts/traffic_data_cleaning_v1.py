import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load raw data
df = pd.read_csv(os.path.join(BASE_DIR, '../data/raw/traffic_smartcity_v1.csv'))

# Cleaning
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')
df = df.dropna()

# Save clean data (NAMANYA DISERAGAMKAN)
os.makedirs(os.path.join(BASE_DIR, '../data/clean'), exist_ok=True)
df.to_csv(os.path.join(BASE_DIR, '../data/clean/traffic_smartcity_clean_v1.csv'), index=False)

print("Data Cleaning Completed")