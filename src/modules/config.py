# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
#CITY_NAME = "Bratislava, Slovakia"
#CENTER_COORDS = None#(48.7208, 21.2575)
#RADIUS_KM = None
CITY_NAME = "Cardiff, Wales"
CENTER_COORDS = (51.5050, -3.1960)
RADIUS_KM = 0.7
N_VEHICLES = 18000
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 4000
TIME_STEP = 10
TIME_WINDOW = 300
DISTANCE_FACTOR = 4.0  # Factor to adjust distance in congestion calculations
CLUSTER_RESOLUTION = 4.0  # Resolution for clustering in connectivity analysis

ATTRACTION_POINT = None#(48.7164, 21.2611)
D_ALTERNATIVES = None#3

# --- QUBO/QA Parameters ---
COMP_TYPE = "hybrid"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"
MIN_CLUSTER_SIZE = 5000 #400,1000,1100
MAX_CLUSTERS = 2
FULL = False


