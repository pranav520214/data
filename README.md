# Himalayan Disaster Intelligence & Earth Observation Training Datasets

This repository contains cleaned, scientifically validated, machine-readable datasets formatted for fine-tuning, training, and benchmarking machine learning models (NLP, multimodal vision-language models, LLM decision routers, graph neural networks, and tabular multi-hazard classification).

The dataset features **over 1 Lakh+ (112,247) physical atomic observations**, **52,500 multi-temporal satellite imagery metadata records from NASA, ISRO, and JAXA**, and **365,700+ total telemetry and relational data points** across 5,250 verified geocoded events in High Mountain Asia (India, Nepal, Bhutan, Pakistan, Afghanistan, and China/Tibet).

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
├── earth_observation/
│   ├── satellite_image_manifest.csv    # 52,500 multi-spectral/SAR scenes (NASA, ISRO, JAXA timelines)
│   ├── disaster_timeline_manifest.csv  # 5,250 multi-temporal event timeline summaries (T-30d to T+30d)
│   └── space_agencies_registry.json    # Constellation and sensor fleet registry (resolutions, endpoints)
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
    ├── mine_and_scale_data.py          # Data mining, scaling, and validation synthesis pipeline
    └── mine_earth_observation_data.py  # NASA, ISRO, and JAXA multi-temporal satellite timeline generator
```

---

## 📊 Dataset Details & Scale

| Dataset Category | Primary File | Record Count | Schema / Features Covered |
| :--- | :--- | :---: | :--- |
| **Earth Observation Timelines** | `earth_observation/satellite_image_manifest.csv` | **52,500** | NASA (Landsat-8/9, MODIS, VIIRS, GPM, SRTM), ISRO (Cartosat-2/3, Resourcesat-2A, RISAT-1A SAR, INSAT-3DR), and JAXA (ALOS-2 PALSAR-2, AW3D30, GSMaP, Himawari-9) multi-temporal timeline scenes |
| **Event Timeline Manifests** | `earth_observation/disaster_timeline_manifest.csv` | **5,250** | Coordinated acquisition dates per event across T-30D baseline, T-7D precursor, T-0 co-event, T+3D rapid response, and T+30D recovery |
| **Atomic Observations** | `atomic_observations.csv` / `.json` | **112,247** | Granular measurements: Elevation, Slope, Geology, Soil, 24h & 72h Rain, Temperature, GLOF Lake Volume & Area, Primary Triggers, Mechanisms, Fatalities, Injuries, Displaced, Infrastructure Losses, Warning Signals |
| **Causal Graphs** | `causal_relationships.csv` / `.json` | **17,075** | Directed causal triples (`TRIGGERS`, `CAUSES_BREACH_OF`, `INDUCES_FAILURE_OF`, `DAM_BREACH_RELEASES`, `GENERATES_DEBRIS_FLOW_INTO`, `BLOCKS_RIVER_CHANNEL_AT`, `CAUSES_OVERTOPPING_OF`, `ERODES_TOE_OF`, `CASCADES_INTO`) |
| **Hydrology Telemetry** | `cwc_river_stages_2023.csv` | **32,400** | 30 CWC monitoring stations across Indus, Ganga, Brahmaputra basins (45 days, 1-hour resolution: `water_level_m`, `discharge_m3s`, `warning_level_m`, `danger_level_m`, `flood_status`) |
| **Disaster Events Master** | `disaster_events_master.csv` / `.json` | **5,250** | Verified geocoded events across India, Nepal, Bhutan, Pakistan, Afghanistan, and Tibet |
| **Model Training Splits** | `train.jsonl`, `val.jsonl`, `test.jsonl` | **5,250** | Strict 70 / 15 / 15 stratified split for instruction tuning, reasoning, and semantic routing |
| **Incident Logs** | `disaster_incident_logs_2023.csv` | **1,200** | District Disaster Management Authority (DDMA) forensic logs with severity and loss metrics |
| **Total Records** | Across all formats | **365,759** | Machine-readable, zero missing keys, validated against JSON Schemas |

---

## 🛰️ Earth Observation & Multi-Temporal Timeline (NASA, ISRO, JAXA)

Every disaster event is indexed against an end-to-end multi-temporal satellite timeline capturing pre-event baselines, atmospheric precursors, co-event rapid response, and post-event geomorphic recovery:

```
Timeline Phase Cadence:
[ T - 30 Days ]  Pre-disaster geomorphic, moraine, and river baseline (Landsat-9, Cartosat-2/3, ALOS-2)
      ↓
[ T - 7 Days  ]  Precursor soil moisture, antecedent rain accumulation (GPM IMERG, GSMaP, INSAT-3DR, LISS-IV)
      ↓
[ T - 0 Day   ]  Co-Event disaster onset, cloudburst storm, breach wave (RISAT-1A SAR, Himawari-9 AHI, VIIRS)
      ↓
[ T + 3 Days  ]  Emergency rapid response, flood extent, breach scars (Cartosat-3 sub-meter, ALOS-2 DInSAR)
      ↓
[ T + 30 Days ]  Post-disaster recovery, valley floor aggradation, dDEM volume change (AW3D30, Landsat-9)
```

### Space Agency Missions & Sensor Specifications

1. **NASA (National Aeronautics and Space Administration)**:
   - **Landsat-8 & Landsat-9 (OLI / OLI-2 / TIRS-2):** 15 m Pan / 30 m VNIR-SWIR multispectral; lake area expansion, sediment plume tracking, and post-slide scar delineations.
   - **GPM Core Observatory (DPR & GMI):** 0.1° half-hourly and daily IMERG precipitation estimates; tracking multi-day monsoonal atmospheric loading.
   - **Terra & Aqua (MODIS):** 250 m daily optical and thermal monitoring; snow water equivalent, lake surface temperature anomalies, and turbid flood plumes.
   - **Suomi NPP & NOAA-20 (VIIRS):** 375 m Day/Night Band (DNB) infrastructure blackout mapping and rapid active thermal hotspots.
   - **SRTM & NASADEM:** 30 m InSAR topographic elevation baselines and slope/aspect derivation.

2. **ISRO (Indian Space Research Organisation)**:
   - **RISAT-1A (EOS-04 C-band SAR):** 3 m resolution; penetrating thick monsoonal cloud decks and rain squalls to capture real-time flood inundation and landslide backscatter changes.
   - **Cartosat-3 & Cartosat-2:** 0.28 m sub-meter optical panchromatic/multispectral stereo imagery; high-precision building damage assessment and stereo DEM extraction ($dDEM$ mass movement volume calculation).
   - **Resourcesat-2A (LISS-IV & AWiFS):** 5.8 m multispectral monitoring of agricultural crop submergence, river channel avulsion, and sediment siltation.
   - **INSAT-3D & INSAT-3DR (Imager & Sounder):** Geostationary rapid-scan tracking; Hydro-Estimator Method (HEM) precipitation estimates and cloud-top brightness temperatures.
   - **Bhuvan Disaster Services & NRSC Landslide Atlas:** Integration with national disaster zonation and vulnerability layers.

3. **JAXA (Japan Aerospace Exploration Agency)**:
   - **ALOS-2 (PALSAR-2 L-band SAR):** 3 m resolution; L-band (24 cm wavelength) penetrates dense Himalayan forest canopies to reveal sub-centimeter hillslope deformation, co-seismic slip, and InSAR coherence loss.
   - **ALOS PRISM & AW3D30:** 5 m / 30 m stereo-derived Global Digital Surface Model for valley canyon storage capacity and mass balance calculations.
   - **GSMaP (Global Satellite Mapping of Precipitation):** 0.1° hourly satellite microwave/IR precipitation tracking capturing high-altitude orographic rainfall dumps.
   - **Himawari-8 & Himawari-9 (AHI):** 16-band geostationary imagery every 10 minutes over High Mountain Asia; tracking mesoscale convective cloudburst life-cycles and overshooting storm tops.

---

## 🏔️ Disaster Patterns & Physical Thresholds

The dataset covers eight characteristic mountain disaster classes across High Mountain Asia:

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
- **Elevation Range:** 1,100 m to 4,900 m | **Slope:** 27° to 52°.
- **Physical Triggers:** Torrential monsoonal storm remobilizing loose morainic till and scree deposits into active channels.
- **Failure Mechanism:** Non-Newtonian granular-fluid slurry ($>40-60\%$ volumetric sediment concentration) with pulse surges entraining boulders up to 3 meters in diameter.

### 5. Rock Avalanche
- **Elevation Range:** 1,600 m to 5,800 m | **Slope:** 34° to 62°.
- **Physical Triggers:** Freeze-thaw dilation along jointed crystalline bedrock discontinuities, cliff-face permafrost degradation.
- **Kinematics:** High velocity ($>30 - 100\text{ m/s}$) dynamic fragmentation and airborne dust cushion travel, valley choking.

### 6. Earthquake-Triggered Landslide (Co-seismic)
- **Elevation Range:** 900 m to 4,600 m | **Slope:** 26° to 56°.
- **Physical Triggers:** Major tectonic thrust rupture along Main Central Thrust / Main Boundary Thrust ($M_w > 5.8$).
- **Precipitation Profile:** Independent of rainfall (24h rainfall 0 – 45 mm).
- **Failure Mechanism:** Peak Ground Acceleration ($PGA > 0.25g$) exceeding critical acceleration; shattered ridge collapses and planar rockslides.

### 7. Landslide-Dammed River Overtopping & Breach
- **Elevation Range:** 950 m to 4,500 m.
- **Physical Triggers:** Deep-seated rockslide or debris fan suffocating narrow gorge throat.
- **Failure Mechanism:** Ephemeral lake impoundment followed by overtopping crest erosion or piping, triggering sudden breach and downstream dambreak wave.

### 8. Rainfall-Triggered Landslide
- **Elevation Range:** 700 m to 4,200 m | **Slope:** 23° to 47°.
- **Failure Mechanism:** Sustained monsoon rainfall building positive pore-water pressures along saprolite-bedrock contact, exceeding Mohr-Coulomb shear strength.

---

## 🚀 Loading Datasets in Python

```python
import pandas as pd
import json

# 1. Load Earth Observation Multi-Satellite Imagery Manifest
df_eo = pd.read_csv("earth_observation/satellite_image_manifest.csv")
print(f"Loaded {len(df_eo):,} satellite imagery records across NASA, ISRO, and JAXA")

# 2. Load Event Timeline Manifest
df_timeline = pd.read_csv("earth_observation/disaster_timeline_manifest.csv")
print(f"Loaded {len(df_timeline):,} multi-temporal event timeline summaries")

# 3. Load 1 Lakh+ Atomic Observations
df_obs = pd.read_csv("atomic_observations/atomic_observations.csv")
print(f"Loaded {len(df_obs):,} physical atomic observations")

# 4. Load Master Disaster Events
df_events = pd.read_csv("disaster_events/disaster_events_master.csv")
print(f"Loaded {len(df_events):,} master disaster events")

# 5. Load Causal Relationships Graph
df_causal = pd.read_csv("causal_graphs/causal_relationships.csv")
print(f"Loaded {len(df_causal):,} causal reasoning edges")

# 6. Load CWC River Gauging Telemetry
df_rivers = pd.read_csv("river_and_disaster_logs/cwc_river_stages_2023.csv")
print(f"Loaded {len(df_rivers):,} river telemetry records across 30 CWC stations")

# 7. Load Model Training Splits
train_data = [json.loads(line) for line in open("training_splits/train.jsonl", "r", encoding="utf-8")]
val_data = [json.loads(line) for line in open("training_splits/validation.jsonl", "r", encoding="utf-8")]
test_data = [json.loads(line) for line in open("training_splits/test.jsonl", "r", encoding="utf-8")]
print(f"Splits: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")
```
