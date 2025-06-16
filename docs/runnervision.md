# RunnerVision

![runnervision_logo](../blueprints/runner_vision/static/images/RunnerVisionLogo_transparent.png)

RunnerVision is a running biomechanic analysis system for analyzing running form using computer vision.

## Overview

RunnerVision uses computer vision (BlazePose) to provide detailed analysis of running biomechanics. The system captures, processes, and visualizes key running form metrics to track improvements and identify potential issues before they lead to injury.

## Demo
![runnervision_demo](../blueprints/runner_vision/static/images/RunnerVision_Demo.gif)

## Features

### 📌 Current Features 
- Multi-angle video capture protocol (side and rear views)
- BlazePose skeleton tracking implementation
- Core running metrics calculation:
  - Key Side Metrics:
    - Foot strike pattern (heel/midfoot/forefoot)
    - Foot landing position relative to center of mass
    - Posture and trunk position
    - Knee angle
  - Key Rear Metrics:
    - Left/Right foot crossover
    - Hip Drop
    - Pelvic Tilt
    - Shoulder Rotation
- Basic report generation with key metrics and figures
- Video output with frame by frame metric values
- Language model driven analysis

### 🚀 Planned Features
- Longitudinal tracking of metrics over time, stored in central database
- Machine learning model for form optimization recommendations
- Comparison to elite runner benchmarks
- Video processing outside of controlled environments (drone capture)
- Automatic detection of fatigue-based form deterioration
- Integration with training log data for performance correlation
- Injury risk assessment based on biomechanical patterns
- Scoring metrics to grade overall form

## Setup Requirements

### Hardware
- 2× smartphone cameras or GoPro with tripods
- Treadmill with consistent lighting
- Computer with GPU support (recommended)

### Software Dependencies
- Python 3.8+
- OpenCV, MediaPipe (BlazePose), TensorFlow/PyTorch
- Pandas, NumPy, Matplotlib/Plotly
- Flask

## Usage

### Data Collection Protocol

#### Camera Positioning:
- Side View: hip height, perpendicular to treadmill, 8-10 feet away
- Rear View: hip height, directly behind treadmill, 6-8 feet away

Use tripods, mark floor positions with tape.

#### Environment:
- Consistent lighting, solid-colored backdrop, same time of day

#### Subject:
- Shirtless or form-fitting shirt, bright shoes

### Recording:
- 60fps
- 60 seconds per pace (easy/moderate/threshold)
- Include countdown and full warm-up/main/cooldown

### Video Export Settings:

**Rear Video:**
- 1080x1920, 59.94 fps, 9:16 aspect

**Side Video:**
- 1920x1080, 60 fps, 16:9 aspect