# SalesIQ: E-Commerce Sales Intelligence

This project is an end-to-end data intelligence and analysis system designed for e-commerce sales tracking. It features an ETL pipeline, an inventory prioritization system based on a min-heap implementation, statistical visualization dashboards, a regional sales map, and a FastAPI backend to expose analytics endpoints.

## Features

- **ETL Pipeline**: Cleans raw, noisy transactional data (handling duplicates, invalid quantity values, currency formatting, discount boundaries, and missing unit price values through product-wise median imputation).
- **Min-Heap Inventory Priority Queue**: Custom min-heap implementation that ranks products based on replenishment priority scores.
- **Data Visualizations**: Generates static Matplotlib-based dashboards showing revenue trends, product distributions, and a heatmap.
- **Geospatial Mapping**: Generates an interactive regional sales distribution map using Folium.
- **REST API Backend**: Fast and lightweight API server using FastAPI that serves cleaned records, summaries, and inventory rankings.

## Project Structure

The project consists of the following key components:
- `solution.py`: The entry point for the ETL pipeline, min-heap builder, dashboard generation, and regional map construction.
- `api.py`: The FastAPI application exposing endpoints for the processed e-commerce datasets.
- `salesiq-dashboard.html`: Front-end web dashboard that visualizes backend data.
- `dashboard_data.json` / `dashboard_data.js`: The serialized data payloads output by the ETL process.
- `requirements.txt`: Python package requirements.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Install Dependencies
Run the following command to install all required libraries:
```bash
pip install -r requirements.txt
```

## Running the Application

### 1. Execute the ETL Pipeline and Analysis
To process raw data and generate updated dashboards and maps, run:
```bash
python solution.py
```
This script performs the full analysis workflow and writes data output files, including `cleaned_orders.csv`, `product_summary.csv`, `sales_dashboard.png`, `regional_sales_map.html`, and `dashboard_data.json`.

### 2. Launch the API Backend Server
To start the FastAPI backend server on local port 8000:
```bash
python -m uvicorn api:app --reload
```
Once started, you can access the interactive API documentation at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- Redoc UI: `http://127.0.0.1:8000/redoc`

### 3. Open the Frontend Dashboard
Open `salesiq-dashboard.html` directly in your web browser. Alternatively, you can serve the directory using a lightweight HTTP server:
```bash
python -m http.server 8080
```
Then navigate to `http://127.0.0.1:8080/salesiq-dashboard.html` in your web browser.
