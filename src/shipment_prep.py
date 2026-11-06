import os
import pandas as pd
os.makedirs("data",exist_ok=True)
df = pd.read_csv("/content/data/hubs.csv")
new_shipments = pd.read_csv("/content/data/new_shipments.csv")
raw_shipments = pd.read_csv("/content/data/raw_shipments.csv")

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

import numpy as np

def clean_shipments(raw, medians=None):
  shipments = raw.drop_duplicates().copy()

  shipments["service_type"] = shipments["service_type"].astype(str).str.strip().str.title()

  shipments["payment_mode"] = shipments["payment_mode"].fillna("Unknown")

  shipments["declared_value_inr"] = pd.to_numeric(shipments["declared_value"].str.replace("₹","",regex=False).str.replace(",","",regex=False),errors="coerce")
  shipments = shipments.drop(columns=["declared_value"])

  shipments["dispatch_time"] = pd.to_datetime(shipments["dispatch_time"],errors="coerce")

  shipments["parcel_weight_kg"] = pd.to_numeric(shipments["parcel_weight_kg"],errors="coerce")
  shipments["parcel_weight_kg"]=shipments["parcel_weight_kg"].where(shipments["parcel_weight_kg"]<=100,np.nan)

  shipments["distance_km"] = pd.to_numeric(shipments["distance_km"], errors="coerce")

  columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr"
    ]
  if medians is None:
        medians = {column: float(shipments[column].median()) for column in columns}
  else:
        medians = {column: float(medians[column]) for column in columns }
  for column in columns:
        shipments[column] = shipments[column].fillna(medians[column])

  return shipments, medians
def build_features(clean,hubs):
 result = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity"
            ]
        ],
        on="hub_id",
        how="left",
        validate="many_to_one")
 result["promised_hours"] = (
        result["service_type"]
        .map(PROMISED_HOURS)
        .astype(int))
 result["is_cod"] = (result["payment_mode"] == "COD").astype(int)

 result["is_weekend"] = (result["dispatch_time"].dt.dayofweek >= 5).astype(int)

 result["dispatch_hour"] = (result["dispatch_time"].dt.hour).astype(int)

 result["log_declared_value"]= np.log(result["declared_value_inr"])

 return result
