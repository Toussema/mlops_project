import requests

url = "http://localhost:8001/predict"
data = {"text": "I need admin access to the database"}

response = requests.post(url, json=data)
print("Status:", response.status_code)
print("Response:", response.json())
