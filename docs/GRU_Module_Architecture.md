# GRU Forecasting Module Architecture

This document provides an intuitive and rigorous understanding of how the GRU-based forecasting model works within the AeroCast system — with symbolic representation, shape transitions, and real-world anchoring.

---

## 1. Symbolic Architecture Diagram

We feed a sequence of `T=10` time steps, each having `F=3` features (temperature, humidity, rainfall), into a single-layer GRU. We extract the last hidden state and pass it through a fully connected layer to generate the final prediction (e.g., next temperature).

### GRU Symbolic Flow:

```
     Input: x = [x₁, x₂, ..., xₜ] ∈ ℝ^{T×F}
                          ↓
                ┌────────────────┐
                │     GRU        │  hₜ⁽ˡ⁾ ∈ ℝ^{H} for t ∈ [1, T]
                │  (1 layer)     │
                └────────────────┘
                          ↓
                 Select h_T⁽ˡ⁾ → last hidden state
                          ↓
              ┌────────────────────────┐
              │ Fully Connected Layer  │  ŷ ∈ ℝ
              │  (hidden → 1 output)   │
              └────────────────────────┘
                          ↓
                  Final Forecast Output
```

- `x_t ∈ ℝ^F`: input at time step `t`
- `h_t^(l) ∈ ℝ^H`: hidden state at layer `l` and time `t`
- `T = 10`, `F = 3`, `H = 64`

---

## 2. Input Feature Matrix: Shape & Visualization

Each input sample is a **sequence of 10 days**, with **3 features** per day:

```
X ∈ ℝ^{10×3} = 
[
 [T₁, H₁, R₁],   ← Day 1 (temp, humidity, rain)
 [T₂, H₂, R₂],   ← Day 2
 ...
 [T₁₀, H₁₀, R₁₀] ← Day 10
]
```

When training in batch:
- Shape = `[batch_size, sequence_length, num_features]`
- Example: `[32, 10, 3]`

---

## 3. GRU Forward Pass Code: Step-by-Step

```python
class GRURegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)              # out ∈ [batch, seq_len, hidden_dim]
        return self.fc(out[:, -1, :])     # pick h_T → shape [batch, output_dim]
```

### Dissection of `out[:, -1, :]`

| Expression        | Meaning                                                   |
|-------------------|-----------------------------------------------------------|
| `:`               | all batch samples                                         |
| `-1`              | final time step (`t = 10`)                                |
| `:`               | all hidden units (e.g., 64)                               |
| Result Shape      | `[batch_size, hidden_dim]` → e.g., `[32, 64]`            |
| After `fc()`      | `[batch_size, 1]` → final temperature prediction          |

---

## 4. Why `.squeeze()` is Used During Training

```python
pred = model(xb).squeeze()
```

- Output before `.squeeze()` = `[32, 1]`
- Target `yb` = `[32]`
- `.squeeze()` → `[32]` → removes the 1-dimension to match target shape

---

## 5. Example Flow (Real Case)

If:
- `SEQ_LEN = 10`
- Features = temperature, humidity, rainfall

Then one input:
```python
X_sample = torch.tensor([[22.0, 78.0, 5.0],
                         [22.1, 76.0, 4.8],
                         ...
                         [23.0, 72.5, 0.2]])  # shape = [10, 3]
```

Final steps:
```python
x_batch = X_sample.unsqueeze(0)             # → [1, 10, 3]
out, _ = self.gru(x_batch)                  # → [1, 10, 64]
last_hidden = out[:, -1, :]                 # → [1, 64]
prediction = self.fc(last_hidden)           # → [1, 1]
```

---

## Summary

| Stage                  | Shape                | Purpose                        |
|------------------------|----------------------|--------------------------------|
| Input Sequence         | `[1, 10, 3]`         | One 10-day input window        |
| GRU Output             | `[1, 10, 64]`        | Hidden states across sequence  |
| Select Last hₜ         | `[1, 64]`            | Capture last time insight      |
| FC Output              | `[1, 1]`             | Predict temperature            |
| `.squeeze()`           | scalar               | Match shape with ground truth  |

---

This symbolic understanding is key to interpreting AeroCast's forecasting logic and aligns with best practices in deploying real-time predictive systems.
