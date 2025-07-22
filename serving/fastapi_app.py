# serving/fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import numpy as np

app = FastAPI(title="AeroCast++ GRU Predictor")

# Dummy model placeholder (will be loaded in Step 2)
model = None

@app.get("/")
def read_root():
    return {"message": "AeroCast++ GRU Predictor is running!"}
