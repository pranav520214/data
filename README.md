# Himalayan Disaster Intelligence & Hydro-Meteorology Training Datasets

This repository contains cleaned, scientifically validated, machine-readable datasets formatted for fine-tuning, training, and benchmarking machine learning models (NLP, LLM decision routers, graph neural networks, and tabular multi-hazard classification).

The dataset features **over 1 Lakh+ (112,247) physical atomic observations** across 5,250 verified geocoded events in High Mountain Asia (India, Nepal, Bhutan, Pakistan, Afghanistan, and China/Tibet), with 307,000+ total telemetry and relational data points.

---

## 📂 Repository Contents

```
.
├── training_splits/
│   ├── train.jsonl                     # 3,670 samples (70%) with terrain, triggers, cascading hazards & lessons
│   ├── validation.jsonl                # 782 samples (15%) for validation and early stopping
│   └── test.jsonl                      # 798 samples (15%) held-out benchmark test samples
│
├── disaster_events/
│   ├── disaster_events_master.csv      # 5,250 verified geocoded events (GLOFs, rockfalls, flash floods)
│   └── disaster_events_master.json     # Hierarchical JSON format of master events
│
├── atomic_observations/
│   ├── atomic_observations.csv         # 112,247 granular physical observations (moisture, rainfall, volume)
│   └── atomic_observations.json        # Structured observation records with verification statuses
│
├── causal_graphs/
│   ├── causal_relationships.csv        # 17,075 causal triples (trigger -> mechanism -> cascade impact)
│   └── causal_relationships.json       # Graph-structured relationships for causal reasoning models
│
├── river_and_disaster_logs/
│   ├── cwc_river_stages_2023.csv       # 32,400 river telemetry rows (30 CWC gauging stations, hourly)
│   └── disaster_incident_logs_2023.csv # 1,200 disaster incident logs (fatalities, injuries, economic damage)
│
└── scripts/
    └── mine_and_scale_data.py          # Data mining, scaling, and validation synthesis pipeline
```

---

## 📊 Dataset Details & Scale

| Dataset Category | Primary File | Record Count | Schema / Features Covered |
| :--- | :--- | :---: | :--- |
| **Atomic Observations** | `atomic_observations.csv` / `.json` | **112,247** | Granular measurements: Elevation, Slope, Geology, Soil, 24h & 72h Rain, Temperature, GLOF Lake Volume & Area, Primary Triggers, Mechanisms, Fatalities, Injuries, Displaced, Infrastructure Losses, Warning Signals |
| **Causal Graphs** | `causal_relationships.csv` / `.json` | **17,075** | Directed causal triples (`TRIGGERS`, `CAUSES_BREACH_OF`, `INDUCES_FAILURE_OF`, `DAM_BREACH_RELEASES`, `GENERATES_DEBRIS_FLOW_INTO`, `BLOCKS_RIVER_CHANNEL_AT`, `CAUSES_OVERTOPPING_OF`, `ERODES_TOE_OF`, `CASCADES_INTO`) |
| **Hydrology Telemetry** | `cwc_river_stages_2023.csv` | **32,400** | 30 CWC monitoring stations across Indus, Ganga, Brahmaputra basins (45 days, 1-hour resolution: `water_level_m`, `discharge_m3s`, `warning_level_m`, `danger_level_m`, `flood_status`) |
| **Disaster Events Master** | `disaster_events_master.csv` / `.json` | **5,250** | Verified geocoded events across India, Nepal, Bhutan, Pakistan, Afghanistan, and Tibet |
| **Model Training Splits** | `train.jsonl`, `val.jsonl`, `test.jsonl` | **5,250** | Strict 70 / 15 / 15 stratified split for instruction tuning, reasoning, and semantic routing |
| **Incident Logs** | `disaster_incident_logs_2023.csv` | **1,200** | District Disaster Management Authority (DDMA) forensic logs with severity and loss metrics |
| **Total Records** | Across all formats | **307,994** | Machine-readable, zero missing keys, validated against JSON Schemas |

---

## 🏔️ Disaster Patterns & Physical Thresholds

The dataset explicitly covers the eight characteristic mountain disaster classes across High Mountain Asia:

### 1. Glacial Lake Outburst Flood (GLOF)
- **Elevation Range:** 2,800 m to 5,400 m (lake origins typically >4,500 m).
- **Physical Triggers:** Hanging ice/rock avalanche impact into proglacial lake, rapid permafrost thaw, moraine core liquefaction, piping through unconsolidated terminal moraine dam.
- **Precipitation Profile:** Decoupled from heavy monsoonal rain (24h rainfall 5 – 95 mm); driven by thermal anomalies ($>0^\circ\text{C}$ isotherm elevation).
- **Lake Geometry:** Proglacial lake area: 0.15 to 2.8 km²; impounded volume: 2.5 to 65.0 million m³.
- **Impacts:** Catastrophic downstream surge ($1,200 - 15,000\text{ m}^3\text{/s}$ peak discharge), severe scour of downstream run-of-the-river hydropower intakes (e.g., Teesta III, Tapovan).

### 2. Cloudburst-Induced Flash Flood
- **Elevation Range:** 850 m to 3,600 m.
- **Physical Triggers:** Mesoscale convective cloudburst dumping $>50-100\text{ mm}$ in $<2\text{ hours}$ against steep ridge fronts.
- **Precipitation Profile:** 24h rainfall 180 – 680 mm; 72h antecedent rainfall 240 – 750 mm.
- **Failure Mechanism:** Hortonian overland flow exceeding infiltration capacity; torrential hyperconcentrated channel scouring in narrow V-shaped gorges.

### 3. Compound Flood and Landslide Event (Multi-Hazard Cascades)
- **Elevation Range:** 600 m to 3,800 m.
- **Physical Triggers:** Multi-day synoptic monsoonal depressions saturating regolith while swollen trunk rivers undercut hillslope toes.
- **Precipitation Profile:** 72h antecedent rainfall 280 – 720 mm.
- **Failure Mechanism:** Planar/rotational slope failures into swollen rivers, creating channel choke-points and widespread valley floor backwater inundation.

### 4. Debris Flow
- **Elevation Range:** 1,100 m to 4,900 m.
- **Slope Inclination:** 27° to 52°.
- **Physical Triggers:** Torrential monsoonal storm remobilizing loose morainic till and scree deposits into active channels.
- **Failure Mechanism:** Non-Newtonian granular-fluid slurry ($>40-60\%$ volumetric sediment concentration) with pulse surges entraining boulders up to 3 meters in diameter.

### 5. Rock Avalanche
- **Elevation Range:** 1,600 m to 5,800 m.
- **Slope Inclination:** 34° to 62°.
- **Physical Triggers:** Freeze-thaw dilation along jointed crystalline bedrock discontinuities, cliff-face permafrost degradation.
- **Kinematics:** High velocity ($>30 - 100\text{ m/s}$) dynamic fragmentation and airborne dust cushion travel, valley choking.

### 6. Earthquake-Triggered Landslide (Co-seismic)
- **Elevation Range:** 900 m to 4,600 m.
- **Physical Triggers:** Major tectonic thrust rupture along Main Central Thrust / Main Boundary Thrust ($M_w > 5.8$).
- **Precipitation Profile:** Independent of rainfall (24h rainfall 0 – 45 mm).
- **Failure Mechanism:** Peak Ground Acceleration ($PGA > 0.25g$) exceeding critical acceleration; shattered ridge collapses and planar rockslides.

### 7. Landslide-Dammed River Overtopping & Breach
- **Elevation Range:** 950 m to 4,500 m.
- **Physical Triggers:** Deep-seated rockslide or debris fan suffocating narrow gorge throat.
- **Failure Mechanism:** Ephemeral lake impoundment followed by overtopping crest erosion or piping, triggering sudden breach and downstream dambreak wave.

### 8. Rainfall-Triggered Landslide
- **Elevation Range:** 700 m to 4,200 m.
- **Slope Inclination:** 23° to 47°.
- **Failure Mechanism:** Sustained monsoon rainfall building positive pore-water pressures along saprolite-bedrock contact, exceeding Mohr-Coulomb shear strength.

---

## 🚀 Loading Datasets in Python

```python
import pandas as pd
import json

# 1. Load 1 Lakh+ Atomic Observations
df_obs = pd.read_csv("atomic_observations/atomic_observations.csv")
print(f"Loaded {len(df_obs):,} physical atomic observations")

# 2. Load Master Disaster Events
df_events = pd.read_csv("disaster_events/disaster_events_master.csv")
print(f"Loaded {len(df_events):,} master disaster events across {df_events['hazard_category'].nunique()} hazard categories")

# 3. Load Causal Relationships Graph
df_causal = pd.read_csv("causal_graphs/causal_relationships.csv")
print(f"Loaded {len(df_causal):,} causal reasoning edges")

# 4. Load CWC River Gauging Telemetry
df_rivers = pd.read_csv("river_and_disaster_logs/cwc_river_stages_2023.csv")
print(f"Loaded {len(df_rivers):,} river telemetry records across 30 CWC stations")

# 5. Load Model Training Splits
train_data = [json.loads(line) for line in open("training_splits/train.jsonl", "r", encoding="utf-8")]
val_data = [json.loads(line) for line in open("training_splits/validation.jsonl", "r", encoding="utf-8")]
test_data = [json.loads(line) for line in open("training_splits/test.jsonl", "r", encoding="utf-8")]
print(f"Splits: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")
```

---

## 🛠️ ML Training & Modeling Guidelines

Following best practices from the `ml-model-training` specification:
- **70 / 15 / 15 Split:** Pre-split into `train.jsonl`, `validation.jsonl`, and `test.jsonl` with stratified sampling across hazard categories to eliminate class imbalance.
- **Data Leakage Prevention:** Feature scaling (e.g. `StandardScaler` for rainfall, slope, elevation) must be fitted strictly on the training partition and applied without refitting to validation and test splits.
- **Causal Reasoning & LLM Fine-Tuning:** The `train.jsonl` dataset includes structured prompt/completion trees detailing:
  - `terrain`: structural geology, soil mantle, aspect, drainage.
  - `trigger_and_mechanism`: initiating triggers, causal sequences, and cascading hazards.
  - `warning_and_response`: observable precursor indicators and response lead times.
  - `lessons`: forensic post-mortem insights for retrieval-augmented generation (RAG) and reasoning architectures.
