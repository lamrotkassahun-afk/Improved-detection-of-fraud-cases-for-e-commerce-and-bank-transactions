import requests
import json

url = "http://127.0.0.1:5000/predict"
data = {
    "purchase_value": 120.0,
    "age": 30,
    "source": "Ads",
    "browser": "Chrome",
    "sex": "M"
}

print(f"Sending request to {url}...")

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    # Pretty-print the JSON response
    print("Response Data:", json.dumps(response.json(), indent=4))
except Exception as e:
    print(f"Connection Error: {e}")
    print("Make sure your api.py is running in another terminal window!")