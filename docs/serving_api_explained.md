# FastAPI Serving: GRU Forecasting Endpoint

This document explains the FastAPI-based inference API for AeroCast++.

---

## Endpoints

### 1. Health Check

```http
GET /
```
Returns:
```json
{
  "message": "AeroCast++ GRU Predictor is running!"
}
```

---

### 2. Forecast Endpoint

```http
POST /predict
```
Accepts a JSON payload with historical sensor values. Returns the next predicted value.

#### Input Schema

```json
{
  "sequence": [[22.3], [22.1], [21.9], [22.0], [22.4]]
}
```

- `sequence`: 2D list representing a time series (shape `[seq_len, 1]`).  
- Values must be floats wrapped in inner lists (each = one feature).

#### Response Schema

```json
{
  "input_length": 5,
  "forecast": 22.08
}
```

- `input_length`: number of timesteps received.  
- `forecast`: model’s predicted next value (float).

---

## cURL Example

```bash
curl -X POST "http://127.0.0.1:8000/predict"      -H "Content-Type: application/json"      -d '{"sequence": [[22.3], [22.1], [21.9], [22.0], [22.4]]}'
```

---

## Internals

- GRU model (`gru_weather_forecaster.pt`) is loaded via PyTorch and set to `eval()`.
- Input → `torch.tensor` of shape `(1, seq_len, 1)`.
- Output → scalar float (rounded before returning).

---

## Swagger UI

Open:
```
http://127.0.0.1:8000/docs
```
Use FastAPI’s interactive docs to test endpoints quickly.

---

## File Location

Implementation lives in:
```
serving/fastapi_app.py
```

---