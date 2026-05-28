from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import traceback


try:
    model = joblib.load("saved_models/eta_model.pkl")

    try:
        node2vec_dict = joblib.load("saved_models/node2vec_embeddings.pkl")
    except FileNotFoundError:
        print("WARNING: node2vec_embeddings.pkl not found. Falling back to dummy vectors for staging.")
        node2vec_dict = {}

except Exception as e:
    print(f"CRITICAL INIT ERROR: Unable to load static object payloads: {str(e)}")
    model = None
    node2vec_dict = {}


app = FastAPI(
    title="Delhivery Network Intelligence Architecture API",
    version="1.2.0"
)


class ETARequest(BaseModel):
    osrm_time: float
    osrm_distance: float
    actual_distance_to_destination: float
    hour: int
    day_of_week: int
    segment_osrm_time: float
    segment_osrm_distance: float
    source_center: str
    destination_center: str


# ROUTE ENDPOINTS
@app.get("/")
def home():
    return {
        "status": "Online",
        "system": "Delhivery Network Intelligence API Core Gateway"
    }

@app.post("/predict_eta")
def predict_eta(data: ETARequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Inference Model Binary File Missing or Corrupt on Core Server Path.")

    try:
    
        feature_dict = {
            "osrm_time": data.osrm_time,
            "osrm_distance": data.osrm_distance,
            "actual_distance_to_destination": data.actual_distance_to_destination,
            "hour": data.hour,
            "day_of_week": data.day_of_week,
            "segment_osrm_time": data.segment_osrm_time,
            "segment_osrm_distance": data.segment_osrm_distance
        }


        src_vector = node2vec_dict.get(data.source_center, np.zeros(64))
        # Handle cases where embedding array might be short or wrong type
        if len(src_vector) != 64:
            src_vector = np.zeros(64)
        
        for i in range(64):
            feature_dict[f"src_emb_{i}"] = float(src_vector[i])


        dst_vector = node2vec_dict.get(data.destination_center, np.zeros(64))
        if len(dst_vector) != 64:
            dst_vector = np.zeros(64)
            
        for i in range(64):
            feature_dict[f"dst_emb_{i}"] = float(dst_vector[i])


        features = pd.DataFrame([feature_dict])


        expected_columns = [
            "osrm_time", "osrm_distance", "actual_distance_to_destination", 
            "hour", "day_of_week", "segment_osrm_time", "segment_osrm_distance"
        ] + [f"src_emb_{i}" for i in range(64)] + [f"dst_emb_{i}" for i in range(64)]
        
        features = features[expected_columns]


        prediction = model.predict(features)[0]

        return {
            "predicted_eta": round(float(prediction), 2),
            "source_centrality": 0.0, 
            "destination_centrality": 0.0
        }

    except Exception as server_exception:
        error_trace = traceback.format_exc()
        print("--- INTERNAL BACKEND ERROR TRACE ---")
        print(error_trace)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Inference Model Error: {str(server_exception)}"
        )