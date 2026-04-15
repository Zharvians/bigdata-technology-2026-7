import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load CLEAN data (bukan raw lagi)
df = pd.read_csv(os.path.join(BASE_DIR, '../data/clean/traffic_smartcity_clean_v1.csv'))

# Feature engineering
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.dayofweek
df['lag1'] = df['traffic'].shift(1)

df = df.dropna()

# Training
x = df[['hour', 'day', 'lag1']]
y = df['traffic']

model = RandomForestRegressor()
model.fit(x, y)

# Save model
os.makedirs(os.path.join(BASE_DIR, '../models'), exist_ok=True)
joblib.dump(model, os.path.join(BASE_DIR, '../models/traffic_model_v1.pkl'))

print("Model Training Completed")