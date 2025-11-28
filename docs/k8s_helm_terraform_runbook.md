# Kubernetes, Helm & Terraform Integration Runbook
**Project:** AeroCast (Polyglot Microservices)

This document serves as a step-by-step reference for setting up the AeroCast infrastructure on a local Kubernetes cluster (Kind), deploying via Helm/Terraform, and executing the "Self-Healing" drift experiment inside the cluster.

---

## 1. Cluster Setup & Image Management

Since Kind (Kubernetes in Docker) cannot see local Docker images by default, we must explicitly tag and load them.

### Step A: Create Cluster
```bash
kind create cluster --name aerocast
```

### Step B: Build & Tag Images

Match the tags defined in `helm/aerocast/values.yaml` (which point to ghcr.io).

```bash
# 1. Build local images (ensure code is fresh)
docker compose build --no-cache

# 2. Tag for Kind/Helm
docker tag aerocast-api:latest ghcr.io/mdgolammafuz/aerocast-api:latest
docker tag aerocast-worker:latest ghcr.io/mdgolammafuz/aerocast-worker:latest
docker tag aerocast-queue:latest ghcr.io/mdgolammafuz/aerocast-queue:latest
```

### Step C: Load Images into Kind

This moves the bits from your local Docker daemon into the Kind cluster nodes.

```bash
kind load docker-image ghcr.io/mdgolammafuz/aerocast-api:latest --name aerocast
kind load docker-image ghcr.io/mdgolammafuz/aerocast-worker:latest --name aerocast
kind load docker-image ghcr.io/mdgolammafuz/aerocast-queue:latest --name aerocast
```

---

## 2. Helm Deployment

### Namespace Conflict Fix

Issue: We initially had a `templates/namespace.yaml` file **and** ran `helm install --create-namespace`.  
This caused: `"namespace already exists"`.

**Fix:** Delete `templates/namespace.yaml` and rely solely on the CLI flag.

### Deploy Command

```bash
helm upgrade --install aerocast ./helm/aerocast   --namespace aerocast   --create-namespace
```

### Verification

Check if the architecture (Core Pod + Queue Pod) spun up correctly.

```bash
# Check Pods
kubectl get pods -n aerocast

# Check Services (Verify Queue is ClusterIP)
kubectl get svc -n aerocast
```

---

## 3. The "Self-Healing" Experiment (In-Cluster Runbook)

In Docker Compose, we ran scripts locally. In Kubernetes, we must execute scripts inside the running containers because they share the internal `/app/data` volume.

### Prerequisite: Set Pod Variable

This variable must be set in every new terminal tab you open.

```bash
export CORE_POD=$(kubectl get pod -n aerocast -l app=aerocast-core -o jsonpath="{.items[0].metadata.name}")
```

---

### Step 1: Clean Slate (The "Quotes" Fix)

Mistake: Running  
`kubectl exec ... rm -rf /app/data/*`  
failed because zsh expanded `*` **on your Mac**.

Fix:

```bash
kubectl exec -n aerocast $CORE_POD -c feeder -- sh -c "rm -rf /app/data/processed/*"
```

---

### Step 2: Generate Calm Data (Bootstrap)

Run the simulator in "Calm" mode for 5 minutes to ensure >24 datapoints (GRU window size).

```bash
kubectl exec -n aerocast $CORE_POD -c feeder -- timeout 300s python streaming/ingestion/heatwave_simulator.py
```

---

### Step 3: Generate CSV & Initial Train

```bash
# 1. Generate CSV
kubectl exec -n aerocast $CORE_POD -c feeder -- python data/generate_training_data_sim.py

# 2. Verify Trainer picked it up
kubectl logs -f -n aerocast $CORE_POD -c trainer
# Look for: "Cold start: Training initial model" -> "Training complete"
```

---

### Step 4: Trigger Heatwave (The Chaos)

Inject anomaly and watch logs in separate terminals.

```bash
# Terminal 1: Drift Logs
kubectl logs -f -n aerocast $CORE_POD -c drift

# Terminal 2: Trainer Logs
kubectl logs -f -n aerocast $CORE_POD -c trainer

# Terminal 3: Trigger Heatwave
kubectl exec -it -n aerocast $CORE_POD -c feeder -- python streaming/ingestion/heatwave_simulator.py
```

Success:  
`"Event posted to queue"` → `"Received event"` → `"Regenerating CSV"`.

---

## 4. Terraform Provisioning (IaC)

Terraform manages the Helm release automatically.

### Configuration (`infra/main.tf`):

- `chart = "../helm/aerocast"`
- `create_namespace = true`
- `wait = false`

### Execution

```bash
cd infra
terraform init
terraform apply
```

Verification:

```bash
kubectl get pods -n aerocast
```

---

## 5. Debugging Lessons Learned

### A. The Zsh Wildcard Error

Error:  
`zsh: no matches found: /app/data/processed/*`  
Cause: Local shell expands `*`.

Fix: Use `sh -c "..."` or escape `\*`.

---

### B. Environment Variable Scope

Error:  
`Error: required argument POD not found`  
Cause: `$CORE_POD` not exported in new tab.

Fix: Re-export variable in every new terminal window.

---

### C. The "EmptyDir" Volume

EmptyDir starts empty on each pod restart.  
Unlike Docker Compose, it **does not mount local Mac files**.

Impact: We must run data generation scripts **inside the pod**.

---

### D. The 422 Validation Error

Cause: Feeder sent `{"features": ...}` while API expected `{"sequence": ...}`.  
Fix: Update feeder code **and rebuild Docker images**.

---

