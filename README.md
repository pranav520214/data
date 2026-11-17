# Geotechnical & Himalayan Disaster Intelligence Open Data Repository

A curated open-data repository providing structured geotechnical telemetry, high-mountain disaster incident catalogs, causal cascade graphs, river monitoring telemetry, and hardware Bill of Materials (BOM) specifications.

---

## 📂 Repository Structure

```
.
├── himalayan_disaster_intelligence/
│   ├── events/
│   │   ├── disaster_events_master.csv       # 1,050 geocoded historical disaster events (GLOFs, rockfalls, cloudbursts)
│   │   └── disaster_events_master.json      # Full JSON metadata with source provenance and terrain metrics
│   ├── observations/
│   │   ├── atomic_observations.csv          # 20,510 granular physical observations (rainfall, moisture, velocity)
│   │   └── atomic_observations.json         # Structured observation records with verification statuses
│   ├── causal_graphs/
│   │   ├── causal_relationships.csv         # 3,029 causal relationship triples linking triggers to cascading hazards
│   │   └── causal_relationships.json        # Machine-readable causal dependency graphs
│   ├── schemas/                             # JSON Schemas for events, observations, and causal relationships
│   ├── quality/                             # Validation reports, duplicate checks, and missing data audits
│   └── sources/                             # Bibliography and official agency data registry
│
├── training_datasets/
│   ├── train.jsonl                          # 735 training samples with full physical mechanisms & lessons learned
│   ├── validation.jsonl                     # 157 validation samples
│   ├── test.jsonl                           # 158 benchmark evaluation samples
│   └── evaluation_metrics.json              # Baseline calibration and classification performance metrics
│
├── river_and_hydrology/
│   ├── cwc_river_stages_2023.csv            # 6,480 river telemetry rows from Central Water Commission (CWC)
│   ├── disaster_incident_logs_2023.csv      # 120 disaster incident records from NDMA (economic damage, casualties)
│   └── earth_observation_manifest.json      # Remote sensing data manifest (Copernicus DEM, ERA5, GPM IMERG)
│
├── sensor_telemetry/
│   ├── sample_telemetry.csv                 # 18-channel calibration telemetry from physical geotechnical slope rig
│   └── landslide_telemetry.db               # SQLite database schema for real-time risk scoring and alert dispatch
│
└── hardware_bom/
    ├── Project_STHIRA_Bill_Of_Materials.xlsx              # Complete multi-sheet BOM, power rails, and wiring matrix
    ├── Water_Cleaning_Boat_Material_List.xlsx             # Hardware requirements for autonomous surface vessel
    └── Water_Cleaning_Boat_Material_List_UserOriginal.xlsx  # Original component procurement worksheet
```

---

## 📊 Dataset Summaries

### 1. Himalayan Disaster Intelligence
- **Coverage:** India (Uttarakhand, Himachal Pradesh, Sikkim, J&K), Nepal, Pakistan, China (Tibet).
- **Hazard Classes:** Glacial Lake Outburst Floods (GLOF), Cloudburst-Induced Flash Floods, Compound Flood & Landslides, Landslide-Dammed River Overtopping, Debris Flows, Earthquake-Triggered Landslides.
- **Physical Attributes:** Digital elevation ($m$), slope inclination ($^\circ$), antecedent 24h & 72h precipitation ($mm$), river basin identification, trigger mechanisms, and lead-time warnings.

### 2. River Stage & Hydro-Meteorology (CWC & NDMA 2023)
- Real-time stage readings across critical river basins monitoring warning levels, danger levels, and highest recorded flood levels (HFL).
- Incident damage assessments tracking displacement, human impact, and local economic loss in Lakh INR.

### 3. Geotechnical Slope Telemetry
- Physical experimental test data recording pore-water progression across slope crest, mid-slope, and slope toe.
- Dual 6-DOF IMU differential inclination, vibration RMS, displacement probe measurements, and operational decision states (`NORMAL`, `WATCH`, `WARNING`, `SLOPE_FAILURE`, `SENSOR_FAULT`).

---

## 📜 Usage & Citations
All datasets are provided for academic research, disaster risk reduction (DRR) modeling, and educational demonstration purposes. When referencing primary source data, consult `himalayan_disaster_intelligence/sources/bibliography.json`.
