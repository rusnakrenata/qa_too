# Configuration constants for the traffic simulation and optimization project

# --- Simulation/City Parameters ---
#CITY_NAME = "Bratislava, Slovakia"
#CENTER_COORDS = None#(48.7208, 21.2575)
#RADIUS_KM = None
CITY_NAME = "Košice, Slovakia"
CENTER_COORDS = (48.7208, 21.2575)
RADIUS_KM = 10
N_VEHICLES = 200
K_ALTERNATIVES = 3
MIN_LENGTH = 500
MAX_LENGTH = 4000
TIME_STEP = 5
TIME_WINDOW = 300

# --- QUBO/QA Parameters ---
COMP_TYPE = "sa"             # 'sa', 'hybrid', or 'qpu'
ROUTE_METHOD = "duration"       # or "distance"


