import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate timestamps for a full month
dates = pd.date_range(start="2026-08-01", end="2026-08-31 23:00:00", freq="h")
corridors = ["Challenge", "Iwo Road", "Bodija", "Ojoo", "Mokola"]

weather_options = ["Clear", "Rain", "Heavy Rain"]
road_types = ["Highway", "Urban", "Junction"]
# Map corridors to broad location zones
location_map = {
    "Challenge": "Central",
    "Iwo Road": "East",
    "Bodija": "West",
    "Ojoo": "North",
    "Mokola": "South"
}

data = []

for date in dates:
    hour = date.hour
    day = date.dayofweek
    for c in corridors:
        if 7 <= hour <= 9 or 16 <= hour <= 19:
            base_vol = np.random.randint(1500, 2500)
            delay = np.random.uniform(1.6, 2.5)
        else:
            base_vol = np.random.randint(400, 1200)
            delay = np.random.uniform(1.0, 1.4)
            
        # Random weather and road type for simulation
        weather = np.random.choice(weather_options, p=[0.75, 0.20, 0.05])
        road_type = np.random.choice(road_types, p=[0.2, 0.7, 0.1])
        location_zone = location_map.get(c, "Central")

        data.append({
            "timestamp": date,
            "corridor_id": c,
            "baseline_volume": base_vol,
            "travel_delay_index": delay,
            "weather": weather,
            "road_type": road_type,
            "location_zone": location_zone
        })

df = pd.DataFrame(data)
df.to_csv("ibadan_traffic_historical.csv", index=False)
print("Sample dataset 'ibadan_traffic_historical.csv' created successfully!")