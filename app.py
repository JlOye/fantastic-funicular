from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the saved pipeline (preprocessor + model)
BASE_DIR = Path(__file__).resolve().parent
pipeline = joblib.load(BASE_DIR / "best_peak_model.pkl")

CORRIDORS = ["Bodija", "Challenge", "Iwo Road", "Mokola", "Ojoo"]
WEATHER_OPTIONS = ["Clear", "Rain", "Heavy Rain"]
ROAD_TYPES = ["Highway", "Urban", "Junction"]
LOCATION_ZONES = ["Central", "East", "West", "North", "South"]

# Corridor-specific alternative route suggestions (example suggestions)
ALTERNATIVE_ROUTES = {
    'Challenge': 'Suggested Alternative: Use Ring Road / Eleyele Bypass or delay departure until 10:00 AM.',
    'Iwo Road': 'Suggested Alternative: Use Old Iyana Church route or delay until 10:00 AM.',
    'Bodija': 'Suggested Alternative: Use Sango-Bodija link road or travel after 11:00 AM.',
    'Ojoo': 'Suggested Alternative: Use Inner-Belt route or delay departure until 10:30 AM.',
    'Mokola': 'Suggested Alternative: Take the University bypass / delay until after 10:00 AM.'
}

# Peak window summaries per corridor (static examples)
PEAK_WINDOWS = {
    'Challenge': {'morning': '7:00 AM – 9:30 AM', 'evening': '4:30 PM – 7:30 PM', 'optimal': '11:00 AM – 3:00 PM'},
    'Iwo Road': {'morning': '7:15 AM – 9:45 AM', 'evening': '4:15 PM – 7:00 PM', 'optimal': '11:00 AM – 3:30 PM'},
    'Bodija': {'morning': '7:00 AM – 9:00 AM', 'evening': '4:00 PM – 7:00 PM', 'optimal': '10:30 AM – 3:30 PM'},
    'Ojoo': {'morning': '6:45 AM – 9:15 AM', 'evening': '4:30 PM – 7:30 PM', 'optimal': '11:00 AM – 3:00 PM'},
    'Mokola': {'morning': '7:30 AM – 9:45 AM', 'evening': '4:45 PM – 7:30 PM', 'optimal': '10:30 AM – 3:00 PM'}
}

METRICS_FILE = BASE_DIR / 'metrics_summary.json'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        hour = int(data.get("hour"))
        day = int(data.get("day"))
        corridor = data.get("corridor")
        weather = data.get("weather")
        road_type = data.get("road_type")
        location_zone = data.get("location_zone")
        
        is_weekend = 1 if day >= 5 else 0
        month = 8
        
        # Determine realistic volume/delay proxies for selected hour and adjust for weather/road type
        if (7 <= hour <= 9) or (16 <= hour <= 19):
            baseline_volume = 2200.0
            travel_delay_index = 2.1
        else:
            baseline_volume = 600.0
            travel_delay_index = 1.1
        
        # Weather adjustments
        if weather == "Rain":
            travel_delay_index += 0.3
            baseline_volume *= 1.05
        elif weather == "Heavy Rain":
            travel_delay_index += 0.6
            baseline_volume *= 1.08
        
        # Road type adjustments
        if road_type == "Highway":
            travel_delay_index -= 0.2
        elif road_type == "Junction":
            travel_delay_index += 0.4
        
        # Build dictionary with exact feature names matching training data
        input_dict = {
            'baseline_volume': [baseline_volume],
            'travel_delay_index': [travel_delay_index],
            'hour_of_day': [hour],
            'day_of_week': [day],
            'is_weekend': [is_weekend],
            'month': [month],
            'corridor_id': [corridor],
            'weather': [weather],
            'road_type': [road_type],
            'location_zone': [location_zone]
        }

        # Convert to DataFrame to retain feature names
        input_df = pd.DataFrame(input_dict)
        
        # Predict using the saved pipeline
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        peak_probability = float(probabilities[1]) * 100
        
        status = "PEAK TRAFFIC PERIOD" if prediction == 1 else "NON-PEAK TRAFFIC PERIOD"
        base_advisory = "Heavy congestion expected along corridor. Consider alternative routes or adjust departure time." if prediction == 1 else "Normal traffic flow expected. Route is clear."

        # Add specific suggestion if peak
        alternative = ALTERNATIVE_ROUTES.get(corridor, "Consider alternative routes or adjust departure time.") if prediction == 1 else ""

        # Peak windows for corridor
        peak_windows = PEAK_WINDOWS.get(corridor, {})

        # Traffic light status uses the two supported congestion levels.
        light = 'red' if peak_probability >= 70 else 'green'

        return jsonify({
            "status": "success",
            "prediction": status,
            "peak_probability": f"{peak_probability:.1f}%",
            "advisory": base_advisory,
            "alternative_suggestion": alternative,
            "peak_windows": peak_windows,
            "traffic_light": light
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        import json
        with open(METRICS_FILE, 'r') as fh:
            data = json.load(fh)
        return jsonify({'status':'success', 'metrics': data})
    except Exception as e:
        return jsonify({'status':'error', 'message': str(e)}), 500


@app.route('/predict-day', methods=['POST'])
def predict_day():
    try:
        data = request.json
        corridor = data.get('corridor')
        day = int(data.get('day'))
        weather = data.get('weather')
        road_type = data.get('road_type')
        location_zone = data.get('location_zone')

        results = []
        input_rows = []
        for hour in range(24):
            is_weekend = 1 if day >= 5 else 0
            month = 8

            # baseline heuristics
            if (7 <= hour <= 9) or (16 <= hour <= 19):
                baseline_volume = 2200.0
                travel_delay_index = 2.1
            else:
                baseline_volume = 600.0
                travel_delay_index = 1.1

            # Weather adjustments
            if weather == 'Rain':
                travel_delay_index += 0.3
                baseline_volume *= 1.05
            elif weather == 'Heavy Rain':
                travel_delay_index += 0.6
                baseline_volume *= 1.08

            # Road type adjustments
            if road_type == 'Highway':
                travel_delay_index -= 0.2
            elif road_type == 'Junction':
                travel_delay_index += 0.4

            input_rows.append({
                'baseline_volume': baseline_volume,
                'travel_delay_index': travel_delay_index,
                'hour_of_day': hour,
                'day_of_week': day,
                'is_weekend': is_weekend,
                'month': month,
                'corridor_id': corridor,
                'weather': weather,
                'road_type': road_type,
                'location_zone': location_zone
            })

        # Batch predict
        import pandas as pd
        df_input = pd.DataFrame(input_rows)
        probs = pipeline.predict_proba(df_input)[:, 1]
        probs_percent = [round(float(p) * 100, 2) for p in probs]

        return jsonify({'status': 'success', 'hourly_probabilities': probs_percent})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)