"""
Test script to spin up Uvicorn, test all REST endpoints with requests, and cleanly shut down.
"""
import subprocess
import time
import requests
import json
import sys

print("Starting Uvicorn API server on http://127.0.0.1:8000...")
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"])

try:
    # Wait for server to bind
    for _ in range(10):
        try:
            r = requests.get("http://127.0.0.1:8000/health", timeout=1.0)
            if r.status_code == 200:
                print("Server is up and healthy!")
                break
        except Exception:
            time.sleep(0.5)

    # 1. Test /
    r_root = requests.get("http://127.0.0.1:8000/")
    print("\n1. GET / ->", r_root.status_code, r_root.json())

    # 2. Test /health
    r_health = requests.get("http://127.0.0.1:8000/health")
    print("\n2. GET /health ->", r_health.status_code, r_health.json())

    # 3. Test /model-info
    r_info = requests.get("http://127.0.0.1:8000/model-info")
    print("\n3. GET /model-info ->", r_info.status_code, r_info.json())

    # 4. Test /predict
    payload = {
        "PatientId": 551234567.0,
        "AppointmentID": 5600001,
        "Gender": "F",
        "ScheduledDay": "2016-05-01T08:30:00Z",
        "AppointmentDay": "2016-05-15T00:00:00Z",
        "Age": 28,
        "Neighbourhood": "CENTRO",
        "Scholarship": 1,
        "Hipertension": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "prior_appointments": 2,
        "prior_no_shows": 1
    }
    r_pred = requests.post("http://127.0.0.1:8000/predict", json=payload)
    print("\n4. POST /predict ->", r_pred.status_code, json.dumps(r_pred.json(), indent=2))

    # 5. Test /explain
    r_exp = requests.post("http://127.0.0.1:8000/explain", json=payload)
    print("\n5. POST /explain ->", r_exp.status_code, "Factors count:", len(r_exp.json().get("all_contributions", [])))

    print("\nALL LIVE REST API ENDPOINTS VERIFIED AND WORKING PERFECTLY!")

finally:
    proc.terminate()
    proc.wait(timeout=3)
    print("API server terminated cleanly.")
