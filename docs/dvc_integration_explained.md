# 📦 DVC Integration in AeroCast++

This document explains how **DVC (Data Version Control)** was integrated into the GRU pipeline in **AeroCast++**, enabling model versioning and large file tracking in a professional, Git-friendly way.

---

## 🚀 Why We Use DVC

- 🔁 Track large files (e.g., `.pt` model files) without polluting Git
- 🧠 Enable experiment reproducibility
- ☁️ Prepare for remote storage (e.g., S3, GDrive, Azure Blob)
- 📊 Integrate with MLflow and pipelines for true MLOps

---

## 🧰 Step-by-Step Integration Summary

### ✅ 1. Initialize DVC

```bash
dvc init
```

Creates `.dvc/`, `.dvcignore`, and adds to Git.

---

### 📦 2. Track the GRU Model

```bash
dvc add artifacts/gru_weather_forecaster.pt
```

This:
- Creates: `artifacts/gru_weather_forecaster.pt.dvc` ✅
- Updates `.gitignore` to ignore the `.pt` file ✅
- Keeps the `.pt` file locally, but Git tracks only the `.dvc` pointer ✅

---

### 🛠️ 3. Remove Git Tracking for the Model

```bash
git rm --cached artifacts/gru_weather_forecaster.pt
```

Then commit:

```bash
git add artifacts/gru_weather_forecaster.pt.dvc artifacts/.gitignore
git commit -m "🔄 Track GRU model with DVC"
```

---

## 🧠 What is `.dvcignore`?

Similar to `.gitignore`, but used by DVC to skip scanning unnecessary files during `dvc add` or pipeline operations.

> Not tracked by Git itself unless manually added.

---

## 📂 Directory Snapshot

```
artifacts/
├── gru_weather_forecaster.pt        # Local model (ignored by Git)
├── gru_weather_forecaster.pt.dvc   # Tracked pointer file
.dvc/
├── config                          # DVC config
.dvcignore                          # DVC ignore rules
.gitignore                          # Git ignore (auto-updated)
```

---

## 🧪 Summary

| Tool | Tracks | Commits to Git? |
|------|--------|-----------------|
| Git | Code, pointer files | ✅ |
| DVC | Heavy data/models    | ❌ (uses `.dvc` pointers) |

> DVC brings **version control to your models**, just like Git does for code.

---

## ✅ Next Steps

- [ ] Connect remote DVC storage (e.g., S3, GDrive)
- [ ] Automate versioning per GRU experiment
- [ ] Link with MLflow run ID for full lineage

---