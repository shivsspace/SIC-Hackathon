from pathlib import Path

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

# Ensure output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)

# File Output Paths
DASHBOARD_FILENAME = str(FRONTEND_DIR / "sales_dashboard.png")
PRODUCT_SUMMARY_FILENAME = str(DATA_DIR / "product_summary.csv")
MAP_FILENAME = str(FRONTEND_DIR / "regional_sales_map.html")
DASHBOARD_JSON_FILENAME = str(FRONTEND_DIR / "dashboard_data.json")

# Algorithm / Processing Parameters
RANDOM_SEED: int = 7
N_RECORDS: int = 600
FRAUD_STD_MULTIPLIER: float = 3.0
DASHBOARD_DPI: int = 150
HOUR_BINS: list[int] = [0, 9, 17, 24]
HOUR_LABELS: list[str] = ["0–8", "9–16", "17–23"]
MARKER_RADIUS_DIVISOR: float = 500_000.0
MAP_CENTER: list[float] = [20.5937, 78.9629]
MAP_ZOOM: int = 5
JITTER_RANGE: float = 1.0

# Geographic Coordinates mapping
REGION_COORDS: dict[str, list[float]] = {
    "North": [28.6139, 77.2090],
    "South": [13.0827, 80.2707],
    "East": [22.5726, 88.3639],
    "West": [19.0760, 72.8777],
    "Other": [20.5937, 78.9629],
}

# Color mapping for Visualizations
PRODUCT_COLORS: dict[str, str] = {
    "Laptop":  "#2196F3",
    "Phone":   "#4CAF50",
    "Tablet":  "#FF9800",
    "Watch":   "#9C27B0",
    "Earbuds": "#F44336",
}
