import pandas as pd
import numpy as np

def load_and_preprocess_traffic_data(file_path):
    df = pd.read_csv(file_path)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['timestamp'].dt.month
    
    corridor_thresholds = df.groupby('corridor_id')['baseline_volume'].transform(lambda x: x.quantile(0.75))
    df['is_peak'] = (df['baseline_volume'] >= corridor_thresholds).astype(int)
    
    # Keep categorical columns (corridor_id, weather, road_type, location_zone) as-is
    # They will be encoded later in a training pipeline using OneHotEncoder
    return df

if __name__ == "__main__":
    processed_df = load_and_preprocess_traffic_data("ibadan_traffic_historical.csv")
    processed_df.to_csv("cleaned_traffic_data.csv", index=False)
    print("Preprocessing completed. Dataset shape:", processed_df.shape)