# Himalayan Disaster Intelligence & Hydro-Meteorology Training Datasets

This repository contains cleaned, machine-readable datasets formatted for fine-tuning, training, and benchmarking machine learning models (NLP, LLM decision routers, graph neural networks, and tabular hazard classification).

---

## 📂 Repository Contents

```
.
├── training_splits/
│   ├── train.jsonl                     # 735 samples with terrain, triggers, cascading hazards & lessons
│   ├── validation.jsonl                # 157 validation samples
│   └── test.jsonl                      # 158 benchmark test samples
│
├── disaster_events/
│   ├── disaster_events_master.csv      # 1,050 verified geocoded events (GLOFs, rockfalls, flash floods)
│   └── disaster_events_master.json     # Hierarchical JSON format of master events
│
├── atomic_observations/
│   ├── atomic_observations.csv         # 20,510 granular physical observations (moisture, rainfall, volume)
│   └── atomic_observations.json        # Structured observation records with verification statuses
│
├── causal_graphs/
│   ├── causal_relationships.csv        # 3,029 causal triples (trigger -> mechanism -> cascade impact)
│   └── causal_relationships.json       # Graph-structured relationships for reasoning models
│
└── river_and_disaster_logs/
    ├── cwc_river_stages_2023.csv       # 6,480 river telemetry rows (water level, discharge, flood status)
    └── disaster_incident_logs_2023.csv # 120 disaster incident logs (fatalities, injuries, economic damage)
```

---

## 📊 Dataset Details

### 1. Model Training Splits (`training_splits/`)
Structured JSONL lines for instruction tuning, reasoning, and semantic routing:
- **`train.jsonl`** (735 records): Detailed prompt/completion and causal reasoning trees.
- **`validation.jsonl`** (157 records) & **`test.jsonl`** (158 records): Isolated splits for evaluation.
- **Schema Fields:** `identification`, `terrain` (elevation, slope, geology, soil), `environmental_conditions` (rainfall 24h/72h, temperature), `trigger_and_mechanism` (primary trigger, causal sequence, cascading hazards), `impacts`, `warning_and_response`, and `lessons`.

### 2. Disaster Events Master (`disaster_events/`)
- **Size:** 1,050 verified events across High Mountain Asia (India, Nepal, Pakistan, China/Tibet).
- **Hazard Classes:** Glacial Lake Outburst Floods (GLOF), Cloudburst-Induced Flash Floods, Compound Flood & Landslides, Landslide-Dammed River Overtopping, Debris Flows, Earthquake-Triggered Landslides.
- **Features:** Latitude, longitude, elevation ($m$), slope inclination ($^\circ$), rainfall 24h & 72h ($mm$), river basin, early warning lead times, fatalities, displaced population.

### 3. Atomic Observations (`atomic_observations/`)
- **Size:** 20,510 individual physical measurement records linked to events by `event_id`.
- **Features:** `event_id`, `observation_id`, `domain`, `factor`, `value`, `unit`, `source`, `observation_date`, `confidence_level`, `verification_status`.

### 4. Causal Graphs (`causal_graphs/`)
- **Size:** 3,029 directed edges.
- **Features:** `relationship_id`, `event_id`, `source_entity`, `relationship_type`, `target_entity`, `causality_certainty`, `evidence_text`.

### 5. Hydrology & Incident Logs (`river_and_disaster_logs/`)
- **`cwc_river_stages_2023.csv`** (6,480 rows): Central Water Commission hourly/daily gauge readings across Himalayan river basins (`water_level_m`, `discharge_m3s`, `warning_level_m`, `danger_level_m`, `highest_flood_level_m`, `flood_status`).
- **`disaster_incident_logs_2023.csv`** (120 rows): National Disaster Management Authority (NDMA) incident reports with severity, human toll, and economic loss metrics.

---

## 🚀 Loading Datasets in Python

```python
import pandas as pd
import json

# 1. Load training split
train_data = [json.loads(line) for line in open("training_splits/train.jsonl", "r", encoding="utf-8")]
print(f"Loaded {len(train_data)} training samples")

# 2. Load events master
df_events = pd.read_csv("disaster_events/disaster_events_master.csv")
print(f"Loaded {len(df_events)} master events")

# 3. Load river telemetry
df_rivers = pd.read_csv("river_and_disaster_logs/cwc_river_stages_2023.csv")
print(f"Loaded {len(df_rivers)} river telemetry records")
```
