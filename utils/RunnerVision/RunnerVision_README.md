# RunnerVision: Biomechanics Analysis Platform

A comprehensive system for analyzing running form using computer vision and wearable data integration.

## Overview

RunnerVision uses computer vision (BlazePose) combined with smartwatch metrics to provide detailed analysis of running biomechanics. The system captures, processes, and visualizes key running form metrics to track improvements and identify potential issues before they lead to injury.

## Features

### 📌 Current Features 
- Multi-angle video capture protocol (side and rear views)
- BlazePose skeleton tracking implementation
- Core running metrics calculation:
  - Foot strike pattern (heel/midfoot/forefoot)
  - Foot landing position relative to center of mass
  - Posture and trunk position
  - Arm carriage and crossover
- Wearable data integration (vertical oscillation, ground contact time, stride length, cadence)
- Basic report generation with key metrics

### 🚀 Planned Features
- Longitudinal tracking of metrics over time
- Advanced metrics:
  - Pelvic drop measurement
  - Hip rotation analysis
  - Knee valgus detection
  - Ankle dorsiflexion angles
  - Push-off power estimation
- Machine learning model for form optimization recommendations
- Comparison to elite runner benchmarks
- Automatic detection of fatigue-based form deterioration
- Real-time feedback capability during treadmill runs
- Integration with training log data for performance correlation
- Injury risk assessment based on biomechanical patterns

## 🛠️ Setup Requirements

### Hardware
- 2× smartphone cameras with tripods
- Treadmill with consistent lighting
- Colored markers for joint position tracking
- Running watches that capture advanced metrics
- Computer with GPU support (recommended)

### Software Dependencies
- Python 3.8+
- OpenCV
- MediaPipe (for BlazePose implementation)
- TensorFlow or PyTorch
- Pandas and NumPy for data processing
- Matplotlib/Plotly for visualization
- Flask for UI (planned)

## Installation

```bash
# Clone the repository
git clone https://github.com/FueledByOat/Runner_Biomechanic_Analysis.git
cd Runner_Biomechanic_Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Collection Protocol

#### Biomechanics Recording Setup
For consistent data collection with phone cameras:

##### Camera Positioning:
- Camera 1 (Side View): Mount at hip height, exactly perpendicular to treadmill, 8-10 feet away

Example Placement:
![alt text](https://github.com/FueledByOat/Runner_Biomechanic_Analysis/blob/main/images/side_cam_distance.jpg "Side Camera setup")

- Camera 2 (Rear View): Mount at hip height, directly behind treadmill, 6-8 feet away

Example Placement:
![alt text](https://github.com/FueledByOat/Runner_Biomechanic_Analysis/blob/main/images/rear_cam_distance.jpg "Rear Camera setup")

- Use tripods with measurements marked on floor with tape for reproducibility
- Mark camera positions with tape on floor


##### Environment Control:
- Consistent lighting (avoid shadows/backlighting)
- Solid-colored backdrop if possible
- Record at same time of day to control for fatigue variables

#### Subject Preparation:
- Shirtless with shorts in contrasting color to skin
- Colored markers (small adhesive dots) at key landmarks:
  - Greater trochanter (hip joint)
  - Lateral knee joint
  - Lateral malleolus (ankle)
  - 5th metatarsal head (outside foot)
  - Acromion process (shoulder)
  - Lateral epicondyle (elbow)
- Consistent bright-colored shoes

### Recording Protocol:
- 60fps minimum (120fps ideal)
- Record 60 seconds at each pace (easy, moderate, threshold)
- Include 10-second countdown before each recording
- Capture complete warm-up, main set, and cool-down sequences

### Video Modification Protocol
Using a tool like OpenShot Video Editor, sample your video then export with the following settings:

#### Rear/Upright Video: 
- Description: FHD Vertical 1080p 59.94 fps
- Width: 1080
- Height: 1920
- FPS: 59.94
- DAR: 9:16
- SAR: 1:1

#### Side/Landscape Video:
- Description: FHD 1080p 60 fps
- Width: 1920
- Height: 1080
- FPS: 60
- DAR: 16:9
- SAR: 1:1

3. Run the collection script:
```bash
python collect_data.py --side_camera 0 --rear_camera 1 --duration 60 --pace easy
```

### 📈 Analysis

```bash
python analyze_run.py --video_path "videos/2025-04-29/side_view.mp4" --watch_data "data/2025-04-29/watch_metrics.csv"
```

### Generating Reports

```bash
python generate_report.py --date "2025-04-29" --output_format "pdf"
```

## 🗄️ Project Structure

```
runner-vision/
├── data/                    # Storage for watch data and processed metrics
├── videos/                  # Raw video footage organized by date
├── processed/               # Processed videos with overlays and annotations
├── reports/                 # Generated analysis reports
├── runner_vision/           # Core Python package
│   ├── collection/          # Video and data collection modules
│   ├── processing/          # Video processing and pose extraction
│   ├── analysis/            # Biomechanics analysis algorithms
│   ├── visualization/       # Plotting and report generation
│   └── models/              # ML models for advanced analysis
├── tools/                   # Utility scripts
├── tests/                   # Test suite
└── notebooks/               # Jupyter notebooks for exploration and testing
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License.
