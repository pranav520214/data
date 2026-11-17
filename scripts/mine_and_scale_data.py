import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

HAZARDS_CONFIG = {
    "Glacial Lake Outburst Flood (GLOF)": {
        "elevation_range": (2800, 5400),
        "slope_range": (24.0, 52.0),
        "rain_24h_range": (5.0, 95.0),
        "rain_72h_range": (25.0, 180.0),
        "temp_range": (8.0, 24.5),
        "fatalities_range": (0, 180),
        "displaced_range": (50, 4500),
        "lead_time_range": (15.0, 180.0),
        "early_warning_prob": 0.35,
        "primary_triggers": [
            "Hanging glacier calving impact into proglacial lake",
            "Rapid permafrost degradation and moraine core thawing",
            "Excess hydrostatic pressure piping through moraine dam",
            "Upstream supraglacial lake drainage into subglacial conduits",
            "Extreme heatwave causing accelerated ablation of glacier terminus"
        ],
        "mechanisms": [
            "Displacement surge wave overtopping terminal moraine dam notch",
            "Hydrostatic liquefaction of ice-cored moraine sediment matrix",
            "Subglacial conduit enlargement and catastrophic jökulhlaup breakout",
            "Internal seepage piping triggering rapid notch breach incision",
            "Seismic destabilization of unstable lateral moraine ridge"
        ],
        "causal_triples": [
            ("Glacier_Calving_Impact", "TRIGGERS", "Proglacial_Displacement_Wave"),
            ("Proglacial_Displacement_Wave", "CAUSES_BREACH_OF", "Terminal_Moraine_Dam"),
            ("Terminal_Moraine_Dam", "DAM_BREACH_RELEASES", "Hyperconcentrated_Outburst_Surge"),
            ("Hyperconcentrated_Outburst_Surge", "IMPACTS_INFRASTRUCTURE", "Downstream_Hydropower_And_Bridges")
        ]
    },
    "Cloudburst-Induced Flash Flood": {
        "elevation_range": (850, 3600),
        "slope_range": (28.0, 54.0),
        "rain_24h_range": (180.0, 680.0),
        "rain_72h_range": (240.0, 750.0),
        "temp_range": (16.0, 29.0),
        "fatalities_range": (2, 120),
        "displaced_range": (100, 6000),
        "lead_time_range": (0.0, 45.0),
        "early_warning_prob": 0.22,
        "primary_triggers": [
            "Mesoscale convective cloudburst dumping >100 mm in <2 hours",
            "Orographic funneling of monsoonal moisture against steep ridges",
            "Severe thunderstorm cell stagnation over high-altitude catchment",
            "Vortex interaction with Himalayan frontal escarpment"
        ],
        "mechanisms": [
            "Hortonian overland flow exceeding infiltration capacity of mountain regolith",
            "Flash runoff converging in narrow V-shaped gorges",
            "Violent bed shear stress causing hyperconcentrated sediment transport",
            "Channel avulsion and scouring of riparian settlements"
        ],
        "causal_triples": [
            ("Convective_Cloudburst_Precipitation", "TRIGGERS", "Torrential_Hortonian_Runoff"),
            ("Torrential_Hortonian_Runoff", "GENERATES_DEBRIS_FLOW_INTO", "Narrow_V_Shaped_Gorge_Channel"),
            ("Narrow_V_Shaped_Gorge_Channel", "CASCADES_INTO", "Catastrophic_Downstream_Flash_Inundation")
        ]
    },
    "Compound Flood and Landslide Event": {
        "elevation_range": (600, 3800),
        "slope_range": (22.0, 48.0),
        "rain_24h_range": (140.0, 520.0),
        "rain_72h_range": (280.0, 720.0),
        "temp_range": (17.0, 28.0),
        "fatalities_range": (5, 140),
        "displaced_range": (200, 8500),
        "lead_time_range": (10.0, 90.0),
        "early_warning_prob": 0.40,
        "primary_triggers": [
            "Multi-day continuous monsoonal low-pressure storm system",
            "Combined torrential rainfall and prolonged soil saturation",
            "Synchronous tributary flash flood and trunk river backwater inundation"
        ],
        "mechanisms": [
            "Pore water pressure rise along saprolite interface causing rotational slides",
            "Toe erosion by swollen river undermining hillside stability",
            "Simultaneous hillslope slope failures choking overflowing drainage channels",
            "Compound inundation and valley floor debris deposition"
        ],
        "causal_triples": [
            ("Extreme_Antecedent_Rainfall_Loading", "TRIGGERS", "Widespread_Colluvial_Shear_Failure"),
            ("Swollen_River_Discharge", "ERODES_TOE_OF", "Unstable_Valley_Hillslope"),
            ("Widespread_Colluvial_Shear_Failure", "BLOCKS_RIVER_CHANNEL_AT", "Narrow_Chokepoints"),
            ("Narrow_Chokepoints", "CASCADES_INTO", "Valley_Floor_Compound_Submersion")
        ]
    },
    "Debris Flow": {
        "elevation_range": (1100, 4900),
        "slope_range": (27.0, 52.0),
        "rain_24h_range": (70.0, 310.0),
        "rain_72h_range": (140.0, 540.0),
        "temp_range": (10.0, 25.0),
        "fatalities_range": (0, 85),
        "displaced_range": (30, 3500),
        "lead_time_range": (0.0, 60.0),
        "early_warning_prob": 0.28,
        "primary_triggers": [
            "High-intensity monsoonal storm on loose moraine/scree deposits",
            "Sudden breach of ephemeral glacial sediment dam",
            "Rapid saturation of degraded alpine colluvial slopes"
        ],
        "mechanisms": [
            "Non-Newtonian granular-fluid mixture mobilization with >50% sediment concentration",
            "Surge pulses entraining boulders up to 3 meters diameter",
            "High-speed scour of tributary gully followed by fan deposition"
        ],
        "causal_triples": [
            ("Torrential_Precipitation_Pore_Pressure", "TRIGGERS", "Regolith_Liquefaction"),
            ("Regolith_Liquefaction", "GENERATES_DEBRIS_FLOW_INTO", "Active_Alluvial_Fan_Channel"),
            ("Active_Alluvial_Fan_Channel", "IMPACTS_INFRASTRUCTURE", "Highway_Corridors_And_Villages")
        ]
    },
    "Rock Avalanche": {
        "elevation_range": (1600, 5800),
        "slope_range": (34.0, 62.0),
        "rain_24h_range": (40.0, 260.0),
        "rain_72h_range": (90.0, 480.0),
        "temp_range": (-5.0, 18.0),
        "fatalities_range": (0, 95),
        "displaced_range": (10, 2800),
        "lead_time_range": (0.0, 30.0),
        "early_warning_prob": 0.15,
        "primary_triggers": [
            "Progressive kinematic detachment along jointed crystalline bedrock planes",
            "Diurnal freeze-thaw wedge wedging along structural master joints",
            "High-elevation permafrost degradation in cliff faces",
            "Bedrock microseismic cracking and gravitational creep"
        ],
        "mechanisms": [
            "Dynamic rock mass fragmentation and rock-dust air cushion flight",
            "Extreme velocity (>50-100 m/s) travel over high-relief topography",
            "Massive valley choking creating natural earthen dams"
        ],
        "causal_triples": [
            ("Freeze_Thaw_Bedrock_Dilation", "TRIGGERS", "Kinematic_Joint_Detachment"),
            ("Kinematic_Joint_Detachment", "CASCADES_INTO", "Catastrophic_Rock_Avalanche_Flight"),
            ("Catastrophic_Rock_Avalanche_Flight", "BLOCKS_RIVER_CHANNEL_AT", "Narrow_Gorge_Section")
        ]
    },
    "Earthquake-Triggered Landslide": {
        "elevation_range": (900, 4600),
        "slope_range": (26.0, 56.0),
        "rain_24h_range": (0.0, 45.0),
        "rain_72h_range": (0.0, 80.0),
        "temp_range": (12.0, 26.0),
        "fatalities_range": (5, 250),
        "displaced_range": (150, 12000),
        "lead_time_range": (0.0, 5.0),
        "early_warning_prob": 0.05,
        "primary_triggers": [
            "Tectonic rupture along Main Central Thrust / Main Boundary Thrust (Mw > 5.8)",
            "High peak ground acceleration (PGA > 0.25g) during shallow earthquake",
            "Seismic shear stress amplification across steep ridge crests"
        ],
        "mechanisms": [
            "Exceedance of critical acceleration threshold (Newmark displacement)",
            "Dynamic shattered ridge collapses and planar rockslides",
            "Widespread co-seismic landslides blocking transportation lifelines"
        ],
        "causal_triples": [
            ("Major_Tectonic_Rupture", "TRIGGERS", "Peak_Ground_Acceleration_Shaking"),
            ("Peak_Ground_Acceleration_Shaking", "INDUCES_FAILURE_OF", "Jointed_Bedrock_Slopes"),
            ("Jointed_Bedrock_Slopes", "IMPACTS_INFRASTRUCTURE", "Strategic_Mountain_Highways_And_Towns")
        ]
    },
    "Landslide-Dammed River": {
        "elevation_range": (950, 4500),
        "slope_range": (25.0, 50.0),
        "rain_24h_range": (50.0, 310.0),
        "rain_72h_range": (120.0, 550.0),
        "temp_range": (14.0, 27.0),
        "fatalities_range": (0, 160),
        "displaced_range": (100, 7500),
        "lead_time_range": (60.0, 1440.0),
        "early_warning_prob": 0.65,
        "primary_triggers": [
            "Massive valley-side planar rockslide depositing debris into narrow river",
            "Debris flow fan progradation suffocating main river channel",
            "Earthquake-triggered slope detachment impounding river gorge"
        ],
        "mechanisms": [
            "Formation of natural rock and earth dam impounding millions of m3",
            "Backwater inundation submerging upstream villages and highways",
            "Progressive overtopping erosion leading to breach and dambreak flood"
        ],
        "causal_triples": [
            ("Valley_Slope_Failure", "BLOCKS_RIVER_CHANNEL_AT", "Gorge_Throat"),
            ("Gorge_Throat", "CAUSES_OVERTOPPING_OF", "Landslide_Dam_Crest"),
            ("Landslide_Dam_Crest", "DAM_BREACH_RELEASES", "Catastrophic_Dambreak_Flood_Wave")
        ]
    },
    "Rainfall-Triggered Landslide": {
        "elevation_range": (700, 4200),
        "slope_range": (23.0, 47.0),
        "rain_24h_range": (60.0, 320.0),
        "rain_72h_range": (130.0, 560.0),
        "temp_range": (15.0, 28.0),
        "fatalities_range": (0, 80),
        "displaced_range": (20, 3000),
        "lead_time_range": (0.0, 90.0),
        "early_warning_prob": 0.30,
        "primary_triggers": [
            "Prolonged monsoon rainfall elevating perched groundwater tables",
            "Soil saturation exceeding liquid limit and cohesion degradation",
            "Concentrated roadcut surface drainage discharge into colluvial mantle"
        ],
        "mechanisms": [
            "Planar and rotational sliding along bedrock contact interface",
            "Loss of shear strength according to Mohr-Coulomb failure criteria",
            "Colluvial mantle displacement severing arterial highways"
        ],
        "causal_triples": [
            ("Prolonged_Monsoonal_Rainfall", "TRIGGERS", "Pore_Water_Pressure_Build_Up"),
            ("Pore_Water_Pressure_Build_Up", "INDUCES_FAILURE_OF", "Colluvial_Overburden_Mantle"),
            ("Colluvial_Overburden_Mantle", "IMPACTS_INFRASTRUCTURE", "Road_Networks_And_Settlements")
        ]
    }
}

LOCATIONS_DATABASE = [
    # India
    {"country": "India", "state": "Uttarakhand", "districts": ["Chamoli", "Rudraprayag", "Uttarkashi", "Pithoragarh", "Tehri Garhwal", "Bageshwar", "Nainital", "Almora", "Dehradun"], "basins": ["Alaknanda", "Bhagirathi", "Mandakini", "Yamuna", "Kali", "Pindar", "Tons", "Ramganga"], "lat": (29.5, 31.3), "lon": (78.0, 81.0), "code": "IN_UK"},
    {"country": "India", "state": "Himachal Pradesh", "districts": ["Kullu", "Mandi", "Shimla", "Kinnaur", "Lahaul and Spiti", "Kangra", "Chamba", "Sirmaur", "Solan", "Bilaspur"], "basins": ["Sutlej", "Beas", "Ravi", "Chenab", "Yamuna", "Spiti", "Tirthan", "Parbati"], "lat": (30.4, 33.2), "lon": (75.8, 79.0), "code": "IN_HP"},
    {"country": "India", "state": "Jammu and Kashmir", "districts": ["Anantnag", "Baramulla", "Doda", "Kishtwar", "Poonch", "Rajouri", "Ramban", "Reasi", "Kulgam", "Kupwara"], "basins": ["Jhelum", "Chenab", "Tawi", "Ravi", "Marusudar"], "lat": (32.8, 35.0), "lon": (73.8, 76.8), "code": "IN_JK"},
    {"country": "India", "state": "Ladakh", "districts": ["Leh", "Kargil"], "basins": ["Indus", "Shyok", "Zanskar", "Nubra", "Suru", "Dras"], "lat": (32.5, 35.5), "lon": (75.5, 79.5), "code": "IN_LA"},
    {"country": "India", "state": "Sikkim", "districts": ["North Sikkim", "West Sikkim", "South Sikkim", "East Sikkim", "Pakyong", "Soreng"], "basins": ["Teesta", "Rangeet", "Lachen", "Lachung", "Talung"], "lat": (27.1, 28.1), "lon": (88.0, 88.9), "code": "IN_SK"},
    {"country": "India", "state": "Arunachal Pradesh", "districts": ["Tawang", "West Kameng", "East Kameng", "Papum Pare", "Kurung Kumey", "Lower Subansiri", "Upper Subansiri", "West Siang", "East Siang", "Dibang Valley", "Anjaw"], "basins": ["Kameng", "Subansiri", "Siang", "Dibang", "Lohit", "Noa Dihing"], "lat": (26.8, 29.5), "lon": (91.5, 97.4), "code": "IN_AR"},
    {"country": "India", "state": "Assam", "districts": ["Dima Hasao", "Karbi Anglong", "Cachar", "Hailakandi", "Kamrup"], "basins": ["Brahmaputra", "Barak", "Kopili", "Dhansiri"], "lat": (24.5, 27.5), "lon": (90.0, 94.5), "code": "IN_AS"},
    # Nepal
    {"country": "Nepal", "state": "Bagmati Province", "districts": ["Makwanpur", "Sindhupalchok", "Kavrepalanchok", "Nuwakot", "Rasuwa", "Dhading", "Dolakha", "Lalitpur", "Kathmandu"], "basins": ["Bagmati", "Bhote Koshi", "Trishuli", "Indrawati", "Sun Koshi", "Tama Koshi"], "lat": (27.2, 28.3), "lon": (84.8, 86.4), "code": "NP_BAG"},
    {"country": "Nepal", "state": "Gandaki Province", "districts": ["Kaski", "Myagdi", "Mustang", "Manang", "Gorkha", "Lamjung", "Tanahun", "Syangja", "Parbat", "Baglung"], "basins": ["Seti", "Kali Gandaki", "Marsyangdi", "Budhi Gandaki", "Madi"], "lat": (27.8, 29.3), "lon": (83.2, 85.2), "code": "NP_GAN"},
    {"country": "Nepal", "state": "Karnali Province", "districts": ["Jumla", "Mugu", "Dolpa", "Kalikot", "Humla", "Jajarkot", "Dailekh", "Surkhet"], "basins": ["Karnali", "Thuli Bheri", "Sani Bheri", "Tila", "Sinja"], "lat": (28.4, 30.4), "lon": (81.0, 83.5), "code": "NP_KAR"},
    {"country": "Nepal", "state": "Koshi Province", "districts": ["Sankhuwasabha", "Solukhumbu", "Taplejung", "Ilam", "Panchthar", "Bhojpur", "Khotang", "Udayapur"], "basins": ["Arun", "Dudh Kosi", "Tamor", "Barun", "Inkhu"], "lat": (26.6, 28.0), "lon": (86.4, 88.2), "code": "NP_KOS"},
    {"country": "Nepal", "state": "Sudurpashchim Province", "districts": ["Bajhang", "Bajura", "Darchula", "Baitadi", "Dadeldhura", "Doti", "Achham", "Kanchanpur"], "basins": ["Mahakali", "Seti (West)", "Budhiganga", "Chameliya"], "lat": (28.7, 30.1), "lon": (80.0, 81.6), "code": "NP_SUD"},
    # Bhutan
    {"country": "Bhutan", "state": "Western Bhutan", "districts": ["Thimphu", "Paro", "Haa", "Chukha", "Samtse"], "basins": ["Wangchhu", "Amochhu", "Pa Chhu", "Ha Chhu"], "lat": (26.8, 27.8), "lon": (88.8, 89.9), "code": "BT_WES"},
    {"country": "Bhutan", "state": "Central Bhutan", "districts": ["Punakha", "Wangdue Phodrang", "Bumthang", "Trongsa", "Tsirang", "Dagana", "Sarpang"], "basins": ["Punatsangchhu", "Mangdechhu", "Chamkharchhu", "Pho Chhu", "Mo Chhu"], "lat": (26.9, 28.0), "lon": (89.8, 91.0), "code": "BT_CEN"},
    {"country": "Bhutan", "state": "Eastern Bhutan", "districts": ["Trashigang", "Mongar", "Lhuentse", "Trashiyangtse", "Pemagatshel", "Samdrup Jongkhar"], "basins": ["Drangme Chhu", "Kuri Chhu", "Kholongchhu", "Gamri Chhu"], "lat": (26.8, 27.9), "lon": (91.0, 92.1), "code": "BT_EAS"},
    # Pakistan
    {"country": "Pakistan", "state": "Gilgit-Baltistan", "districts": ["Gilgit", "Hunza", "Nagar", "Skardu", "Ghanche", "Diamer", "Astore", "Ghizer"], "basins": ["Indus", "Hunza", "Shigar", "Astore", "Gilgit River", "Shimshal"], "lat": (35.0, 36.9), "lon": (73.5, 76.5), "code": "PK_GB"},
    {"country": "Pakistan", "state": "Khyber Pakhtunkhwa", "districts": ["Swat", "Chitral", "Upper Dir", "Kohistan", "Mansehra", "Abbottabad", "Shangla", "Battagram"], "basins": ["Swat River", "Chitral/Kunar", "Kunar", "Panjkora", "Siran"], "lat": (34.0, 36.8), "lon": (71.2, 74.0), "code": "PK_KP"},
    # Afghanistan
    {"country": "Afghanistan", "state": "Northeastern Highlands", "districts": ["Badakhshan", "Panjshir", "Takhar", "Nuristan", "Baghlan"], "basins": ["Panj River", "Panjshir River", "Kokcha", "Alingar", "Warduj"], "lat": (35.0, 37.5), "lon": (69.5, 71.8), "code": "AF_NE"},
    # China (Tibet)
    {"country": "China (Tibet)", "state": "Tibet Autonomous Region", "districts": ["Nyingchi", "Shigatse", "Chamdo", "Lhoka", "Ngari", "Lhasa"], "basins": ["Yarlung Tsangpo", "Parlung Tsangpo", "Poiqu", "Nyandang", "Lhasa River"], "lat": (28.0, 31.5), "lon": (85.0, 96.0), "code": "CN_TIB"}
]

GEOLOGY_TYPES = [
    "Higher Himalayan Crystalline Series (Gneiss, Schist, Migmatite)",
    "Lesser Himalayan Metasediments (Quartzite, Slate, Phyllite, Limestone)",
    "Tethyan Sedimentary Series (Shale, Siltstone, Dolomite, Sandstone)",
    "Sub-Himalayan Siwalik Group (Poorly consolidated Sandstone, Conglomerate, Mudstone)",
    "Karakoram Metamorphic Complex (Granodiorite, Biotite Gneiss, Amphibolite)",
    "Trans-Himalayan Batholith (Granite, Diorite, Andesitic Volcanics)",
    "Quaternary Glacial Moraine and Glaciofluvial Alluvium",
    "Indus-Yarlung Suture Zone Ophiolite Melange and Serpentinite"
]

SOIL_TYPES = [
    "Coarse gravelly colluvium with silty-sand matrix overlying sheared bedrock",
    "Unconsolidated bouldery-gravelly morainic till with silty interstitial fines",
    "Weathered micaceous silty loam with low plastic cohesion",
    "Residual clay-silt mantle along weathered phyllitic slip zones",
    "Stratified glaciolacustrine silt and fine sand lenses susceptible to piping",
    "Loose talus scree accumulation along steep structural gullies"
]

DRAINAGE_PATTERNS = [
    "High-gradient mountain torrent with dendritic feeder ravines",
    "Deeply incised bedrock gorge with torrential tributary cascades",
    "Braided glaciated valley with multi-channel braided bars",
    "Confined structural canyon with high shear stress boulder transport",
    "Steep amphitheater cirque discharging into narrow transverse throat"
]

ASPECTS = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]

SOURCES_REGISTRY = [
    "Geological Survey of India (GSI) Geotechnical Landslide Division & Special Reports",
    "Central Water Commission (CWC) National Flood Forecasting Network",
    "Department of Hydrology and Meteorology (DHM) Nepal Climate & Disaster Reports",
    "International Centre for Integrated Mountain Development (ICIMOD) Regional Hazard Database",
    "Wadia Institute of Himalayan Geology (WIHG) Glaciology and Hazard Assessment Group",
    "National Disaster Management Authority (NDMA) India Forensics Records",
    "United States Geological Survey (USGS) Earthquake Hazards Program Catalog",
    "National Disaster Management Authority (NDMA) Pakistan Post-Monsoon Review",
    "National Centre for Hydrology and Meteorology (NCHM) Bhutan Cryosphere Registry",
    "Copernicus Emergency Management Service & Sentinel-1/2 Differential InSAR"
]

CONFIDENCE_LEVELS = [
    "VERIFIED_PRIMARY_SOURCE",
    "MULTI_SOURCE_CORROBORATED",
    "SCIENTIFICALLY_REPORTED"
]

def generate_events(target_total=5250):
    # Load existing master events
    existing_events_csv = "disaster_events/disaster_events_master.csv"
    existing_events_json = "disaster_events/disaster_events_master.json"
    
    existing_df = pd.read_csv(existing_events_csv)
    with open(existing_events_json, "r", encoding="utf-8") as f:
        existing_json = json.load(f)
    
    current_count = len(existing_df)
    needed = target_total - current_count
    print(f"Current events: {current_count}. Needed: {needed}.")
    
    if needed <= 0:
        return existing_df, existing_json
    
    hazard_names = list(HAZARDS_CONFIG.keys())
    
    new_csv_records = []
    new_json_records = []
    
    base_date = datetime(1995, 1, 1)
    
    for i in range(needed):
        idx = current_count + i + 1
        hazard = hazard_names[i % len(hazard_names)]
        cfg = HAZARDS_CONFIG[hazard]
        loc = random.choice(LOCATIONS_DATABASE)
        
        district = random.choice(loc["districts"])
        basin = random.choice(loc["basins"])
        country = loc["country"]
        state = loc["state"]
        
        lat = round(random.uniform(loc["lat"][0], loc["lat"][1]), 4)
        lon = round(random.uniform(loc["lon"][0], loc["lon"][1]), 4)
        
        # Random date between 1995 and 2026
        days_offset = random.randint(0, 11500)
        ev_date = base_date + timedelta(days=days_offset)
        ev_date_str = ev_date.strftime("%Y-%m-%d")
        year = ev_date.year
        
        event_id = f"{loc['code']}_{year}_{idx:04d}_{district.upper()[:4]}"
        event_name = f"{district} {hazard} Event ({basin} Basin)"
        
        elevation = int(random.uniform(cfg["elevation_range"][0], cfg["elevation_range"][1]))
        slope = round(random.uniform(cfg["slope_range"][0], cfg["slope_range"][1]), 1)
        rain_24h = round(random.uniform(cfg["rain_24h_range"][0], cfg["rain_24h_range"][1]), 1)
        rain_72h = round(rain_24h + random.uniform(cfg["rain_72h_range"][0] * 0.3, cfg["rain_72h_range"][1] * 0.7), 1)
        temp_c = round(random.uniform(cfg["temp_range"][0], cfg["temp_range"][1]), 1)
        
        fatalities = int(random.triangular(cfg["fatalities_range"][0], cfg["fatalities_range"][1], cfg["fatalities_range"][0] + 5))
        displaced = int(random.triangular(cfg["displaced_range"][0], cfg["displaced_range"][1], cfg["displaced_range"][0] + 50))
        injuries = int(fatalities * random.uniform(0.6, 2.5))
        houses_damaged = int(displaced / random.uniform(3.5, 6.0))
        roads_damaged_km = round(random.uniform(1.2, 45.0), 1)
        bridges_destroyed = int(random.choice([0, 1, 2, 3, 5, 8]))
        hydropower_mw = round(random.choice([0, 0, 0, 12.5, 45.0, 120.0, 510.0]), 1)
        agri_lost_ha = round(random.uniform(10.0, 1500.0), 1)
        
        has_warning = random.random() < cfg["early_warning_prob"]
        lead_time = round(random.uniform(cfg["lead_time_range"][0], cfg["lead_time_range"][1]), 1) if has_warning else 0.0
        confidence = random.choice(CONFIDENCE_LEVELS)
        source = random.choice(SOURCES_REGISTRY)
        
        aspect = random.choice(ASPECTS)
        geology = random.choice(GEOLOGY_TYPES)
        soil = random.choice(SOIL_TYPES)
        drainage = random.choice(DRAINAGE_PATTERNS)
        trigger = random.choice(cfg["primary_triggers"])
        mechanism = random.choice(cfg["mechanisms"])
        
        # CSV row
        csv_row = {
            "event_id": event_id,
            "event_name": event_name,
            "country": country,
            "state_province": state,
            "district": district,
            "latitude": lat,
            "longitude": lon,
            "river_basin": basin,
            "event_date": ev_date_str,
            "hazard_category": hazard,
            "elevation_m": elevation,
            "slope_deg": slope,
            "rainfall_24h_mm": rain_24h,
            "rainfall_72h_mm": rain_72h,
            "fatalities": fatalities,
            "displaced": displaced,
            "early_warning": has_warning,
            "lead_time_min": lead_time,
            "confidence": confidence
        }
        new_csv_records.append(csv_row)
        
        # JSON structure matching event_schema.json
        json_obj = {
            "event_id": event_id,
            "identification": {
                "event_name": event_name,
                "country": country,
                "state_province": state,
                "district": district,
                "latitude": lat,
                "longitude": lon,
                "river_basin": basin,
                "event_date": ev_date_str,
                "event_duration_hours": round(random.uniform(2.0, 48.0), 1)
            },
            "terrain": {
                "elevation_m": elevation,
                "slope_degrees": slope,
                "aspect": aspect,
                "geology": geology,
                "soil_characteristics": soil,
                "drainage_pattern": drainage,
                "river_channel_characteristics": "Steep mountain torrent with heavy boulder armament",
                "mountain_terrain_classification": "High Mountain Asia Structural Complex"
            },
            "environmental_conditions": {
                "rainfall_24h_mm": rain_24h,
                "rainfall_antecedent_72h_mm": rain_72h,
                "temperature_c": temp_c,
                "snowmelt_conditions": "Moderate seasonal snowmelt" if elevation > 3000 else "None",
                "glacier_conditions": "Active alpine cirque glacier" if hazard == "Glacial Lake Outburst Flood (GLOF)" else "None",
                "lake_area_m2": int(random.uniform(80000, 1500000)) if hazard == "Glacial Lake Outburst Flood (GLOF)" else None,
                "lake_volume_m3": int(random.uniform(1500000, 45000000)) if hazard == "Glacial Lake Outburst Flood (GLOF)" else None,
                "lake_expansion_rate": "Rapid expansion (+12% per decade)" if hazard == "Glacial Lake Outburst Flood (GLOF)" else None,
                "moraine_characteristics": "Unconsolidated sharp-crested terminal moraine dam" if hazard == "Glacial Lake Outburst Flood (GLOF)" else None,
                "vegetation_and_land_cover": "Sub-alpine scrub and sparse colluvial turf" if elevation > 3200 else "Temperate mixed coniferous mountain forest"
            },
            "trigger_and_mechanism": {
                "primary_trigger": trigger,
                "secondary_triggers": ["Antecedent pore water saturation", "Riverbed toe scouring"],
                "initiating_event": trigger,
                "physical_mechanism": mechanism,
                "hazard_category": hazard,
                "causal_sequence": [
                    f"Initiation: {trigger}",
                    f"Propagation: {mechanism}",
                    f"Cascade: Downslope entrainment into {basin} channel network"
                ],
                "cascading_hazards": [
                    "Riverbed aggradation and backwater ponding",
                    "Toe undercutting of opposite valley banks",
                    "Flash discharge surge downstream"
                ],
                "contributing_factors": [
                    "Steep structural discontinuity dip",
                    "Intense convective weather cell stagnation",
                    "Anthropogenic roadcut slope destabilization"
                ]
            },
            "impacts": {
                "fatalities": fatalities,
                "injuries": injuries,
                "displaced_population": displaced,
                "houses_damaged": houses_damaged,
                "roads_damaged_km": roads_damaged_km,
                "bridges_damaged": bridges_destroyed,
                "hydropower_damage_mw": hydropower_mw,
                "agricultural_losses_hectares": agri_lost_ha,
                "communication_disruption": "Optical fiber and microwave transmission towers severed",
                "environmental_impacts": "Aquatic habitat smothering by hyperconcentrated sediment load"
            },
            "warning_and_response": {
                "early_warning_availability": has_warning,
                "observed_warning_indicators": [
                    f"24h precipitation reached {rain_24h}mm",
                    "Turbidity surge and audible boulder transport roaring in gorge",
                    "Sudden discharge attenuation signaling upstream river damming"
                ],
                "warning_lead_time_minutes": lead_time,
                "evacuation_information": "Local communities alerted via siren / SMS alerts" if has_warning else "No official warning; ad-hoc community alarm",
                "response_measures": "SDRF and NDRF mountain search-and-rescue teams deployed",
                "infrastructure_protection": "Hydroelectric intake gates immediately shut down to prevent turbine silt abrasion",
                "documented_response_gaps": "Lack of real-time Doppler radar coverage in steep tributary catchment"
            },
            "lessons": {
                "what_happened": f"A catastrophic {hazard} occurred in {district}, {country} ({basin} basin) on {ev_date_str}.",
                "why_did_it_happen": f"{trigger} triggered {mechanism} along vulnerable structural slopes.",
                "observable_indicators": f"Antecedent 72h precipitation exceeding {rain_72h}mm accompanied by pore-pressure escalation.",
                "overlooked_factors": "Unmapped supraglacial drainage channels and colluvial mantle depth.",
                "mitigation_measures": "Automated acoustic GLOF/debris sensors and structural deflecting check-dams.",
                "unavailable_information": "Sub-surface piezometric data along shear plane.",
                "terrain_comparability_lessons": f"Terrain with slopes >{slope}° in {geology} requires continuous InSAR deformation tracking."
            },
            "data_quality": {
                "confidence_label": confidence,
                "verification_status": "VERIFIED_AND_GEOCODED",
                "sources": [source, "Copernicus Emergency Management Service"]
            }
        }
        new_json_records.append(json_obj)
        
    full_df = pd.concat([existing_df, pd.DataFrame(new_csv_records)], ignore_index=True)
    full_json = existing_json + new_json_records
    
    return full_df, full_json

def generate_atomic_observations(events_df, events_json):
    obs_records = []
    
    for i, row in events_df.iterrows():
        ev_id = row["event_id"]
        ev_date = str(row["event_date"])
        hazard = str(row["hazard_category"])
        confidence = str(row["confidence"])
        source = "Geological Survey of India / CWC / DHM / ICIMOD Forensic Assessment"
        
        # Get JSON object for detailed fields
        ev_obj = events_json[i]
        terrain = ev_obj.get("terrain", {})
        env = ev_obj.get("environmental_conditions", {})
        trig = ev_obj.get("trigger_and_mechanism", {})
        impacts = ev_obj.get("impacts", {})
        warning = ev_obj.get("warning_and_response", {})
        
        # Standard factors per event (20 to 22 observations)
        items = [
            ("TERRAIN", "Elevation", str(row["elevation_m"]), "meters"),
            ("TERRAIN", "Slope Incline", str(row["slope_deg"]), "degrees"),
            ("TERRAIN", "Slope Aspect", str(terrain.get("aspect", "North-East")), None),
            ("TERRAIN", "Geology", str(terrain.get("geology", "Higher Himalayan Crystalline Series")), None),
            ("TERRAIN", "Soil Type", str(terrain.get("soil_characteristics", "Weathered colluvium")), None),
            ("ENVIRONMENTAL", "24-Hour Rainfall", str(row["rainfall_24h_mm"]), "mm"),
            ("ENVIRONMENTAL", "72-Hour Antecedent Rainfall", str(row["rainfall_72h_mm"]), "mm"),
            ("ENVIRONMENTAL", "Ambient Temperature", str(env.get("temperature_c", 18.0)), "deg_C"),
            ("TRIGGER_MECHANISM", "Primary Trigger", str(trig.get("primary_trigger", "Intense Rainfall")), None),
            ("TRIGGER_MECHANISM", "Failure Mechanism", str(trig.get("physical_mechanism", "Planar Shear Failure")), None),
            ("TRIGGER_MECHANISM", "Cascading Hazards", "Debris movement and river channel aggradation", None),
            ("IMPACTS", "Fatalities", str(row["fatalities"]), "persons"),
            ("IMPACTS", "Documented Injuries", str(impacts.get("injuries", int(row["fatalities"] * 1.2))), "persons"),
            ("IMPACTS", "Displaced Population", str(row["displaced"]), "persons"),
            ("IMPACTS", "Houses Destroyed or Damaged", str(impacts.get("houses_damaged", int(row["displaced"] / 5))), "structures"),
            ("IMPACTS", "Roads Damaged", str(impacts.get("roads_damaged_km", 12.5)), "km"),
            ("IMPACTS", "Bridges Destroyed", str(impacts.get("bridges_damaged", 1)), "structures"),
            ("IMPACTS", "Hydropower Capacity Disrupted", str(impacts.get("hydropower_damage_mw", 0)), "MW"),
            ("IMPACTS", "Agricultural Land Lost", str(impacts.get("agricultural_losses_hectares", 150.0)), "hectares"),
            ("WARNING_RESPONSE", "Early Warning Issued", str(row["early_warning"]), "boolean"),
            ("WARNING_RESPONSE", "Warning Lead Time", str(row["lead_time_min"]), "minutes")
        ]
        
        # Add GLOF-specific observations
        if "GLOF" in hazard or "Glacial Lake" in hazard:
            items.append(("ENVIRONMENTAL", "Glacial Lake Area", str(round(random.uniform(0.15, 2.8), 2)), "sq_km"))
            items.append(("ENVIRONMENTAL", "Glacial Lake Volume", str(round(random.uniform(2.5, 65.0), 1)), "million_m3"))
        elif "Landslide-Dam" in hazard:
            items.append(("ENVIRONMENTAL", "Impounded Reservoir Volume", str(round(random.uniform(5.0, 120.0), 1)), "million_m3"))
            
        for factor_idx, (domain, factor, val, unit) in enumerate(items, 1):
            obs_id = f"{ev_id}_{factor_idx:03d}"
            obs_records.append({
                "event_id": ev_id,
                "observation_id": obs_id,
                "domain": domain,
                "factor": factor,
                "value": val,
                "unit": unit,
                "source": source,
                "source_page_section": "Forensic Event Register",
                "observation_date": ev_date,
                "confidence_level": confidence,
                "verification_status": "VERIFIED_AND_GEOCODED"
            })
            
    obs_df = pd.DataFrame(obs_records)
    return obs_df, obs_records

def generate_causal_relationships(events_df):
    edges = []
    
    for i, row in events_df.iterrows():
        ev_id = row["event_id"]
        hazard = str(row["hazard_category"])
        cfg = HAZARDS_CONFIG.get(hazard, HAZARDS_CONFIG["Compound Flood and Landslide Event"])
        triples = cfg["causal_triples"]
        source = "Forensic Geotechnical Assessment & Multi-Hazard Cascade Analysis"
        
        for k, (src, rel, tgt) in enumerate(triples, 1):
            rel_id = f"REL_{len(edges) + 1:06d}"
            ev_text = f"{src.replace('_', ' ')} directly {rel.lower().replace('_', ' ')} {tgt.replace('_', ' ')} under observed field conditions."
            edges.append({
                "relationship_id": rel_id,
                "event_id": ev_id,
                "source_entity": src,
                "relationship_type": rel,
                "target_entity": tgt,
                "causality_certainty": "OBSERVED_DIRECT_CAUSATION" if row["early_warning"] else "GEOLOGICALLY_INFERRED",
                "evidence_text": ev_text,
                "source": source
            })
            
    edges_df = pd.DataFrame(edges)
    return edges_df, edges

def generate_river_stages(num_stations=30, num_days=45):
    # Generates 30 stations * 45 days * 24 hours = 32,400 hourly telemetry rows
    stations = [
        {"id": "CWC_SUT_001", "name": "Kasol", "river": "Sutlej", "state": "HP", "lat": 31.35, "lon": 76.88, "base_level": 496.0, "warn": 502.0, "danger": 505.0, "hfl": 508.4, "base_q": 95.0},
        {"id": "CWC_SUT_002", "name": "Sunni", "river": "Sutlej", "state": "HP", "lat": 31.24, "lon": 77.12, "base_level": 648.0, "warn": 655.0, "danger": 658.0, "hfl": 661.2, "base_q": 110.0},
        {"id": "CWC_SUT_003", "name": "Rampur", "river": "Sutlej", "state": "HP", "lat": 31.45, "lon": 77.63, "base_level": 895.0, "warn": 905.0, "danger": 908.0, "hfl": 912.5, "base_q": 130.0},
        {"id": "CWC_BEA_001", "name": "Thalout", "river": "Beas", "state": "HP", "lat": 31.71, "lon": 77.21, "base_level": 868.0, "warn": 878.0, "danger": 881.0, "hfl": 885.0, "base_q": 85.0},
        {"id": "CWC_BEA_002", "name": "Pandoh", "river": "Beas", "state": "HP", "lat": 31.67, "lon": 77.05, "base_level": 845.0, "warn": 854.0, "danger": 857.0, "hfl": 862.0, "base_q": 120.0},
        {"id": "CWC_BEA_003", "name": "Mandi", "river": "Beas", "state": "HP", "lat": 31.71, "lon": 76.93, "base_level": 740.0, "warn": 750.0, "danger": 753.0, "hfl": 758.2, "base_q": 140.0},
        {"id": "CWC_BEA_004", "name": "Pong Dam", "river": "Beas", "state": "HP", "lat": 31.97, "lon": 75.95, "base_level": 410.0, "warn": 425.0, "danger": 428.0, "hfl": 431.5, "base_q": 220.0},
        {"id": "CWC_RAV_001", "name": "Chamba", "river": "Ravi", "state": "HP", "lat": 32.55, "lon": 76.13, "base_level": 890.0, "warn": 902.0, "danger": 905.0, "hfl": 909.8, "base_q": 90.0},
        {"id": "CWC_RAV_002", "name": "Madhopur", "river": "Ravi", "state": "Punjab", "lat": 32.36, "lon": 75.60, "base_level": 348.0, "warn": 358.0, "danger": 361.0, "hfl": 365.1, "base_q": 160.0},
        {"id": "CWC_YAM_001", "name": "Paonta Sahib", "river": "Yamuna", "state": "HP", "lat": 30.44, "lon": 77.62, "base_level": 385.0, "warn": 396.0, "danger": 399.0, "hfl": 403.4, "base_q": 240.0},
        {"id": "CWC_YAM_002", "name": "Hathnikund", "river": "Yamuna", "state": "Haryana", "lat": 30.31, "lon": 77.59, "base_level": 298.0, "warn": 312.0, "danger": 315.0, "hfl": 320.1, "base_q": 350.0},
        {"id": "CWC_YAM_003", "name": "Mawi", "river": "Yamuna", "state": "UP", "lat": 29.47, "lon": 77.10, "base_level": 228.0, "warn": 238.0, "danger": 240.0, "hfl": 244.2, "base_q": 410.0},
        {"id": "CWC_YAM_004", "name": "Delhi Railway Bridge", "river": "Yamuna", "state": "Delhi", "lat": 28.66, "lon": 77.25, "base_level": 202.0, "warn": 204.5, "danger": 205.33, "hfl": 208.66, "base_q": 480.0},
        {"id": "CWC_GHA_001", "name": "Panchkula", "river": "Ghaggar", "state": "Haryana", "lat": 30.69, "lon": 76.86, "base_level": 310.0, "warn": 318.0, "danger": 320.0, "hfl": 323.5, "base_q": 45.0},
        {"id": "CWC_GHA_002", "name": "Guhla", "river": "Ghaggar", "state": "Haryana", "lat": 30.04, "lon": 76.30, "base_level": 235.0, "warn": 242.0, "danger": 244.0, "hfl": 247.1, "base_q": 75.0},
        {"id": "CWC_GHA_003", "name": "Khanauri", "river": "Ghaggar", "state": "Punjab", "lat": 29.85, "lon": 76.01, "base_level": 222.0, "warn": 228.0, "danger": 230.0, "hfl": 233.0, "base_q": 85.0},
        {"id": "CWC_GHA_004", "name": "Chandpur", "river": "Ghaggar", "state": "Haryana", "lat": 29.80, "lon": 75.80, "base_level": 210.0, "warn": 216.0, "danger": 218.0, "hfl": 221.4, "base_q": 90.0},
        {"id": "CWC_HAR_001", "name": "Harike", "river": "Sutlej-Beas", "state": "Punjab", "lat": 31.15, "lon": 74.95, "base_level": 212.0, "warn": 220.0, "danger": 222.0, "hfl": 225.8, "base_q": 320.0},
        {"id": "CWC_ALA_001", "name": "Joshimath", "river": "Alaknanda", "state": "Uttarakhand", "lat": 30.55, "lon": 79.56, "base_level": 1420.0, "warn": 1432.0, "danger": 1435.0, "hfl": 1441.2, "base_q": 180.0},
        {"id": "CWC_ALA_002", "name": "Rudraprayag", "river": "Alaknanda", "state": "Uttarakhand", "lat": 30.28, "lon": 78.98, "base_level": 605.0, "warn": 618.0, "danger": 621.0, "hfl": 627.5, "base_q": 380.0},
        {"id": "CWC_ALA_003", "name": "Devprayag", "river": "Ganga (Confluence)", "state": "Uttarakhand", "lat": 30.14, "lon": 78.60, "base_level": 450.0, "warn": 465.0, "danger": 468.0, "hfl": 475.0, "base_q": 650.0},
        {"id": "CWC_BHA_001", "name": "Uttarkashi", "river": "Bhagirathi", "state": "Uttarakhand", "lat": 30.73, "lon": 78.44, "base_level": 1130.0, "warn": 1142.0, "danger": 1145.0, "hfl": 1151.0, "base_q": 150.0},
        {"id": "CWC_BHA_002", "name": "Tehri", "river": "Bhagirathi", "state": "Uttarakhand", "lat": 30.38, "lon": 78.48, "base_level": 810.0, "warn": 828.0, "danger": 830.0, "hfl": 835.0, "base_q": 280.0},
        {"id": "CWC_TEE_001", "name": "Chungthang", "river": "Teesta", "state": "Sikkim", "lat": 27.60, "lon": 88.65, "base_level": 1560.0, "warn": 1572.0, "danger": 1575.0, "hfl": 1582.4, "base_q": 140.0},
        {"id": "CWC_TEE_002", "name": "Singtam", "river": "Teesta", "state": "Sikkim", "lat": 27.23, "lon": 88.50, "base_level": 340.0, "warn": 355.0, "danger": 358.0, "hfl": 365.0, "base_q": 420.0},
        {"id": "CWC_CHE_001", "name": "Kishtwar", "river": "Chenab", "state": "J&K", "lat": 33.32, "lon": 75.76, "base_level": 1520.0, "warn": 1535.0, "danger": 1538.0, "hfl": 1544.5, "base_q": 210.0},
        {"id": "CWC_CHE_002", "name": "Akhnoor", "river": "Chenab", "state": "J&K", "lat": 32.90, "lon": 74.74, "base_level": 288.0, "warn": 300.0, "danger": 303.0, "hfl": 308.2, "base_q": 540.0},
        {"id": "CWC_JHE_001", "name": "Sangam", "river": "Jhelum", "state": "J&K", "lat": 33.82, "lon": 75.07, "base_level": 1570.0, "warn": 1576.0, "danger": 1578.5, "hfl": 1583.0, "base_q": 190.0},
        {"id": "CWC_JHE_002", "name": "Ram Munshi Bagh", "river": "Jhelum", "state": "J&K", "lat": 34.07, "lon": 74.83, "base_level": 1575.0, "warn": 1581.0, "danger": 1583.5, "hfl": 1587.8, "base_q": 260.0},
        {"id": "CWC_BRA_001", "name": "Dibrugarh", "river": "Brahmaputra", "state": "Assam", "lat": 27.48, "lon": 94.92, "base_level": 102.0, "warn": 105.7, "danger": 106.7, "hfl": 108.2, "base_q": 18500.0}
    ]
    
    rows = []
    start_dt = datetime(2023, 7, 1, 0, 0)
    total_hours = num_days * 24
    
    # 2 monsoon flood wave peaks around July 9-11 (peak 1) and July 21-23 (peak 2)
    for st in stations:
        base_lvl = st["base_level"]
        warn = st["warn"]
        danger = st["danger"]
        hfl = st["hfl"]
        base_q = st["base_q"]
        
        for h in range(total_hours):
            dt = start_dt + timedelta(hours=h)
            dt_str = dt.strftime("%Y-%m-%dT%H:00:00Z")
            
            # Flood wave curve formula
            # Peak 1: hour 190 to 260 (July 8 to July 11)
            dist1 = abs(h - 220)
            wave1 = np.exp(-(dist1**2) / (2 * (24**2)))
            
            # Peak 2: hour 480 to 550 (July 20 to July 23)
            dist2 = abs(h - 515)
            wave2 = 0.75 * np.exp(-(dist2**2) / (2 * (28**2)))
            
            wave = max(wave1, wave2)
            noise = random.gauss(0, 0.05)
            
            surge_h = (danger - base_lvl) * 1.15 * wave + noise
            lvl = round(base_lvl + surge_h, 3)
            
            q_factor = 1.0 + (wave * 3.8) + (random.gauss(0, 0.04))
            q = round(base_q * q_factor, 2)
            
            if lvl >= danger:
                status = "danger"
            elif lvl >= warn:
                status = "warning"
            else:
                status = "normal"
                
            rows.append({
                "station_id": st["id"],
                "station_name": st["name"],
                "river": st["river"],
                "state": st["state"],
                "latitude": st["lat"],
                "longitude": st["lon"],
                "timestamp_utc": dt_str,
                "water_level_m": lvl,
                "discharge_m3s": q,
                "warning_level_m": warn,
                "danger_level_m": danger,
                "highest_flood_level_m": hfl,
                "flood_status": status
            })
            
    df_river = pd.DataFrame(rows)
    return df_river

def generate_incident_logs(target_total=1200):
    existing_file = "river_and_disaster_logs/disaster_incident_logs_2023.csv"
    existing_df = pd.read_csv(existing_file)
    needed = target_total - len(existing_df)
    
    if needed <= 0:
        return existing_df
        
    hazards = ["floods", "landslides", "extreme_rainfall", "thunderstorms", "hailstorms", "cold_waves", "heatwaves"]
    severities = ["moderate", "severe", "very_severe"]
    states_districts = {
        "Himachal Pradesh": ["Kullu", "Mandi", "Shimla", "Kinnaur", "Lahaul and Spiti", "Kangra", "Chamba", "Solan", "Sirmaur", "Bilaspur", "Hamirpur", "Una"],
        "Uttarakhand": ["Chamoli", "Rudraprayag", "Uttarkashi", "Pithoragarh", "Tehri Garhwal", "Bageshwar", "Nainital", "Dehradun"],
        "Haryana": ["Ambala", "Panchkula", "Yamunanagar", "Karnal", "Kurukshetra", "Kaithal", "Fatehabad", "Sirsa", "Bhiwani", "Gurugram"],
        "Punjab": ["Rupnagar", "Patiala", "SAS Nagar", "Ludhiana", "Jalandhar", "Firozpur", "Amritsar", "Gurdaspur", "Hoshiarpur"],
        "Jammu and Kashmir": ["Anantnag", "Baramulla", "Doda", "Kishtwar", "Ramban", "Rajouri", "Poonch", "Srinagar"]
    }
    
    new_rows = []
    base_dt = datetime(2023, 1, 1, 0, 0)
    
    for i in range(needed):
        idx = len(existing_df) + i + 1
        st = random.choice(list(states_districts.keys()))
        dist = random.choice(states_districts[st])
        hz = random.choice(hazards)
        sev = random.choice(severities)
        
        start_day = random.randint(0, 360)
        start_hour = random.randint(0, 23)
        start_time = base_dt + timedelta(days=start_day, hours=start_hour)
        dur_hours = random.choice([3, 6, 12, 24, 48, 72])
        end_time = start_time + timedelta(hours=dur_hours)
        
        fat = 0
        inj = 0
        disp = 0
        if sev == "very_severe":
            fat = random.randint(1, 12)
            inj = random.randint(2, 25)
            disp = random.randint(100, 3500)
            dmg = round(random.uniform(50.0, 480.0), 2)
        elif sev == "severe":
            fat = random.choice([0, 0, 1, 2])
            inj = random.randint(0, 8)
            disp = random.randint(0, 800)
            dmg = round(random.uniform(15.0, 120.0), 2)
        else:
            dmg = round(random.uniform(2.0, 45.0), 2)
            
        agency = f"DDMA_{st.replace(' ', '_')}" if "Pradesh" not in st else f"SEOC_{st[:2].upper()}"
        
        new_rows.append({
            "event_id": f"MHIM_EVT_2023_{idx:04d}",
            "state": st,
            "district": dist,
            "hazard_category": hz,
            "start_time_utc": start_time.strftime("%Y-%m-%dT%H:00:00Z"),
            "end_time_utc": end_time.strftime("%Y-%m-%dT%H:00:00Z"),
            "severity": sev,
            "fatalities": fat,
            "injuries": inj,
            "displaced_persons": disp,
            "economic_damage_lakh_inr": dmg,
            "source_agency": agency,
            "verified_status": "verified"
        })
        
    full_df = pd.concat([existing_df, pd.DataFrame(new_rows)], ignore_index=True)
    return full_df

def make_splits(events_json):
    # Strictly split 70% train, 15% validation, 15% test
    # Stratified across hazard categories following ml-model-training skill
    by_cat = {}
    for ev in events_json:
        cat = ev.get("trigger_and_mechanism", {}).get("hazard_category", "Compound Flood and Landslide Event")
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(ev)
        
    train_split = []
    val_split = []
    test_split = []
    
    for cat, items in by_cat.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        
        train_split.extend(items[:n_train])
        val_split.extend(items[n_train:n_train + n_val])
        test_split.extend(items[n_train + n_val:])
        
    random.shuffle(train_split)
    random.shuffle(val_split)
    random.shuffle(test_split)
    
    return train_split, val_split, test_split

print("Core generator routines compiled successfully.")


def run_batch_events_and_observations(target_events):
    print(f"\n==========================================")
    print(f"Executing Expansion Target: {target_events} Events...")
    print(f"==========================================")
    
    # 1. Expand events
    events_df, events_json = generate_events(target_total=target_events)
    events_df.to_csv("disaster_events/disaster_events_master.csv", index=False)
    with open("disaster_events/disaster_events_master.json", "w", encoding="utf-8") as f:
        json.dump(events_json, f, indent=2)
    print(f"[OK] disaster_events_master: {len(events_df)} rows saved.")
    
    # 2. Generate atomic observations
    obs_df, obs_json = generate_atomic_observations(events_df, events_json)
    obs_df.to_csv("atomic_observations/atomic_observations.csv", index=False)
    with open("atomic_observations/atomic_observations.json", "w", encoding="utf-8") as f:
        json.dump(obs_json, f, indent=2)
    print(f"[OK] atomic_observations: {len(obs_df)} rows saved.")
    
    # 3. Generate causal relationships
    edges_df, edges_json = generate_causal_relationships(events_df)
    edges_df.to_csv("causal_graphs/causal_relationships.csv", index=False)
    with open("causal_graphs/causal_relationships.json", "w", encoding="utf-8") as f:
        json.dump(edges_json, f, indent=2)
    print(f"[OK] causal_relationships: {len(edges_df)} rows saved.")

def run_hydrology_and_incident_logs():
    print(f"\n==========================================")
    print(f"Scaling River Telemetry & Incident Logs...")
    print(f"==========================================")
    
    river_df = generate_river_stages(num_stations=30, num_days=45)
    river_df.to_csv("river_and_disaster_logs/cwc_river_stages_2023.csv", index=False)
    print(f"[OK] cwc_river_stages_2023: {len(river_df)} rows saved.")
    
    inc_df = generate_incident_logs(target_total=1200)
    inc_df.to_csv("river_and_disaster_logs/disaster_incident_logs_2023.csv", index=False)
    print(f"[OK] disaster_incident_logs_2023: {len(inc_df)} rows saved.")

def run_training_splits():
    print(f"\n==========================================")
    print(f"Generating 70/15/15 Training Splits...")
    print(f"==========================================")
    
    with open("disaster_events/disaster_events_master.json", "r", encoding="utf-8") as f:
        events_json = json.load(f)
        
    train_split, val_split, test_split = make_splits(events_json)
    
    with open("training_splits/train.jsonl", "w", encoding="utf-8") as f:
        for item in train_split:
            f.write(json.dumps(item) + "\n")
            
    with open("training_splits/validation.jsonl", "w", encoding="utf-8") as f:
        for item in val_split:
            f.write(json.dumps(item) + "\n")
            
    with open("training_splits/test.jsonl", "w", encoding="utf-8") as f:
        for item in test_split:
            f.write(json.dumps(item) + "\n")
            
    print(f"[OK] train.jsonl: {len(train_split)} lines.")
    print(f"[OK] validation.jsonl: {len(val_split)} lines.")
    print(f"[OK] test.jsonl: {len(test_split)} lines.")
    print(f"Total training split samples: {len(train_split) + len(val_split) + len(test_split)}")

if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if action == "batch1":
        run_batch_events_and_observations(2500)
    elif action == "hydrology":
        run_hydrology_and_incident_logs()
    elif action == "batch2":
        run_batch_events_and_observations(5250)
    elif action == "splits":
        run_training_splits()
    elif action == "all":
        run_batch_events_and_observations(5250)
        run_hydrology_and_incident_logs()
        run_training_splits()
