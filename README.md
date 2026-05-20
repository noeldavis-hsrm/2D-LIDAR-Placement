# 2D-LiDAR Sensor Placement & Validation

Python tools for computing, visualising, and empirically validating the optimal placement of 2D-LiDAR sensors in arbitrary indoor environments.

Accompanies the publications:
- D'Avis & Faquiri. *Optimal 2D-LiDAR-Sensor Coverage of a Room.* iWOAR 2024, Potsdam.
- D'Avis & Faquiri. *Optimization and Validation of a 2D-LiDAR-Coverage Algorithm.* PETRA 2025, Corfu.

---

## Overview

The algorithm divides a room into a grid and decomposes each cell into four directional triangle faces (TOP, RIGHT, BOTTOM, LEFT). A greedy placement optimisation selects the minimal set of sensor positions along the room walls such that every reachable, non-marginal triangle is observed from the geometrically correct direction. This directional coverage criterion reduces shadow effects and supports multi-sensor redundancy.

Two obstacle categories are distinguished throughout all tools:

- **Category A** — signal-blocking obstacles (e.g. furniture). LiDAR rays are occluded; a one-cell safety margin is applied around them. Drawn by click-and-drag in the GUI.
- **Category B** — structurally unreachable areas (e.g. built-in wardrobes, pillars). Excluded from the coverage metric entirely; rays pass through them transparently. Drawn by Shift + click-and-drag.

---

## Repository Contents
 
| File | Description |
|---|---|
| `lidar_placement.py` | Interactive GUI for room configuration, obstacle placement, algorithm execution, and JSON export of optimised sensor positions. |
| `lidar_viewer.py` | Live viewer that overlays theoretical coverage with real-time RPLidar scan data. Loads `sensor_positions.json` and assigns COM ports via a Tkinter dialogue. |
| `lidar_validation.py` | Validation tool that accumulates real sensor observations per grid cell and exports a triangle-level coverage heatmap and CSV report. |
 
The repository also includes the sensor position configuration used in the experiments (`sensor_positions.json`) as well as three validation result CSV files recorded on 13 March 2026 in a 3.0 × 3.5 m room at grid cell sizes of 10 cm.
 
---

## Requirements

```
pip install matplotlib numpy pyserial rplidar-roboticia
```

Python 3.9 or later is recommended. The GUI backend uses **TkAgg**; a display (or virtual framebuffer) is required.

---

## Usage

### 1 — Compute sensor placement

```bash
python lidar_placement.py
```

1. Enter room dimensions (width, height in metres) and grid cell size, then click **Apply**.
2. Draw Category-A obstacles by click-and-drag; hold **Shift** to draw Category-B areas.
3. Click **Run Algorithm** to compute the optimal sensor positions.
4. Click **Export JSON** to save `sensor_positions.json` for use with the viewer and validation tool.

### 2 — Live viewer

```bash
python lidar_viewer.py
```

Load `sensor_positions.json` via the file dialogue, assign the COM port of each RPLidar sensor, and confirm. The Matplotlib window displays the theoretical coverage overlaid with live scan data from all connected sensors.

### 3 — Validation

```bash
python lidar_validation.py
```

Load the same `sensor_positions.json`, assign COM ports, and start the measurement. While the sensors are running, the tool accumulates how many distinct sensors observe each grid cell. After stopping, a heatmap is rendered (0 sensors = red, 1 sensor = orange, ≥ 2 sensors = green) and a timestamped CSV is exported.

---

## CSV Format

Each validation CSV contains a header block with run metadata followed by one row per triangle:

```
ci, cj, direction, x_m, y_m, covered_by_sensor, result
```

`covered_by_sensor` is the index of the sensor that covers the triangle in the theoretical model (0-indexed); `result` is the sensor label (S1, S2, …) observed during the real measurement.

---

## Hardware

Tested with the **RPLidar A1M8** (maximum range 6.0 m, 360° field of view). The sensor range constant `SENSOR_RANGE` in all three scripts can be adjusted for other models.


---

## Authors

**Noel D'Avis** and **Silvia Faquiri**  
RheinMain University of Applied Sciences, Wiesbaden, Germany  
