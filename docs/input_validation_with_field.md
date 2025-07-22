# 🛡️ Input Validation with `Field` (Pydantic) in AeroCast++

We use **Pydantic** to validate incoming JSON for the `/predict` endpoint. `Field()` lets us enforce constraints and auto-generate docs in Swagger.

---

## 🎯 Why `Field()`?

- Ensures data integrity *before* inference
- Adds constraints (min/max length, required fields)
- Provides rich descriptions & examples visible in Swagger UI
- Reduces runtime errors in the GRU forward pass

---

## 🧱 Base Structure

```python
from pydantic import BaseModel, Field
from typing import List

class WeatherInput(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        min_items=5,
        max_items=100,
        description="Time series as [[val1], [val2], ...] (shape: seq_len x 1)",
        example=[[23.1], [22.8], [23.0], [22.7], [22.9]]
    )
```

### ✅ Key Parts

| Argument      | Meaning                                               |
|---------------|-------------------------------------------------------|
| `...`         | Field is **required**                                 |
| `min_items`   | Minimum allowed sequence length                       |
| `max_items`   | Maximum allowed sequence length                       |
| `description` | Shown in Swagger UI                                   |
| `example`     | Pre-populates Swagger “Try it out” box                |

---

## 🔍 Validation Example

**Valid input:**
```json
{
  "sequence": [[23.1], [22.8], [23.0], [22.7], [22.9]]
}
```

**Invalid input (too short):**
```json
{
  "sequence": [[23.1]]
}
```
**Swagger / FastAPI response:**
```json
{
  "detail": [
    {
      "loc": ["body", "sequence"],
      "msg": "ensure this value has at least 5 items",
      "type": "value_error.list.min_items",
      "ctx": {"limit_value": 5}
    }
  ]
}
```

---

## 🧪 Where It’s Used

See:
```
serving/fastapi_app.py
```
Model: `WeatherInput`

---

## 🧠 Mental Model

> `BaseModel` gives structure.  
> `Field()` gives rules + docs.  
> FastAPI + Swagger turn both into a live, validated API.

---