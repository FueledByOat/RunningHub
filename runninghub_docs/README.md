# RunningHub Analytics

A modular, open-source platform for runners to analyze and visualize their training data, track strength workouts, evaluate biomechanics, and receive personalized insights via integrated AI tools.

## 📚 Table of Contents
- [Overview](#overview)
- [Core Modules](#core-modules)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Installation](#installation)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license)

## 🚀 Core Modules
- [RunnerVision](docs/runnervision.md) – Biomechanical analysis from video
- [RunStrong](docs/runstrong.md) – Strength training tracking and planning
- [Coach G](docs/coachg.md) – LLM-powered training advice
- [RunningHub Core](docs/runninghub.md) – Fitness metrics, visualizations, dashboards

## 🧪 Demo
![runninghub_demo](./blueprints/running_hub/static/images/RunningHub_Demo.gif)

## 🧱 Technology Stack

| Feature           | Current Tool(s)                                        |
| ----------------- | ------------------------------------------------------ |
| Web Interface     | Flask                                                  |
| Dashboards        | Dash, Javascript, HTML                                 |
| Data Storage      | SQLite                                                 |
| Metrics Analysis  | pandas, scipy, plotly, numpy                           |
| ML Models         | scikit-learn, ruptures, xgboost                        |
| LLM Summaries     | transformers                                           |
| Data Ingestion    | Custom Pipeline (Python)                               |
| Containerization  | Docker, Docker Compose (planned)                       |
| DevOps Monitoring | Grafana + Prometheus (planned)                         |

## 🛠️ Getting Started

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/RunningHub.git
cd RunningHub
```

### 2. Set Up the Environment
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Launch the App
```bash
python app.py  # or however you start your Flask server
```

## 🤝 Contribution Guidelines

* Fork the repo and set up Docker locally.
* Submit well-documented pull requests.
* Open issues for bugs, feature requests, or ideas.

## 📜 License

MIT License

---

Feel free to contribute or adapt this hub for your own running or coaching needs!