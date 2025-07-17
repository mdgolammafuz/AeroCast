
# 🔁 GRU: Data Sequence Construction Logic

This document explains how we construct input and target sequences for training our GRU weather forecasting model.

---

## 📦 Step-by-Step Logic

We start with a processed DataFrame `df`:

```python
df = pd.read_parquet("data/processed/")
features = ["temperature", "humidity", "rainfall"]
X = df[features].values  # shape: [num_days, 3]
```

Let's say:
- `X` has shape `[365, 3]` — 365 days, 3 features per day
- We set `SEQ_LEN = 10` → 10 time steps per sequence

---

### 🧠 Understanding the Sequence Creation

```python
X_seq.append(data[i:i+seq_len])        # Input sequence
y_seq.append(data[i+seq_len][0])       # Target (next day's temperature)
```

---

### 📊 Visual Breakdown

Suppose `data` is:

| i | temperature | humidity | rainfall |
|---|-------------|----------|----------|
| 0 |     20      |   70     |    1     |
| 1 |     21      |   72     |    0     |
| 2 |     19      |   75     |    0     |
| 3 |     18      |   77     |    1     |
| 4 |     22      |   70     |    0     |
| 5 |     23      |   68     |    0     |
| 6 |     24      |   67     |    0     |
| 7 |     25      |   65     |    1     |
| 8 |     26      |   64     |    0     |
| 9 |     27      |   63     |    1     |
|10 |     28      |   61     |    0     |

- `data[0:10]` → Picks rows : index 0 to 9 = input sequence `X_seq[0]` → shape: `[10, 3]`
- `data[10][0]` → Picks row index 5 -> temperature on day 10 = target value `y_seq[0]`

---

## 🎯 Why `data[i+seq_len][0]`?

- It gives the **next day’s temperature** after the sequence
- `[0]` selects the **first feature**, i.e., temperature
- We assume temperature is the first in the feature list

---

## 🔢 Summary Table

| Variable | Shape          | Description                         |
|----------|----------------|-------------------------------------|
| `X_tensor` | `[N, 10, 3]`    | N sequences of 10 days, 3 features |
| `y_tensor` | `[N]`          | N next-day temperatures             |

---

## 🔍 About `idx` in `__getitem__`

```python
def __getitem__(self, idx):
    return self.X[idx], self.y[idx]
```

- `idx` is **just a variable name**, not a keyword
- It’s the **index of the batch** PyTorch asks for
- We could write `def __getitem__(self, i):` or `def __getitem__(self, index):`

---

## ✅ Final Takeaways

- GRU learns to predict `temperature[t+1]` from a 10-day sliding window
- Each training sample is: `[X[i:i+10]] → [temperature at i+10]`
- PyTorch uses `__getitem__` and `__len__` to fetch training batches automatically

