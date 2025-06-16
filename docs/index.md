# RunningHub Analytics

A modular, open-source platform for runners to analyze and visualize their training data, track strength workouts, evaluate biomechanics, and receive personalized insights via integrated AI tools.

## 📚 Table of Contents
<!-- - [Overview](#overview) -->
- [Core Modules](#core-modules)
- [Demo](#demo)
- [Technology Stack](#technology-stack)
<!-- - [Features](#features)
- [Installation](#installation)
- [Contribution Guidelines](#contribution-guidelines)
- [License](#license) -->

## 🚀 Core Modules
- [RunnerVision](runnervision.md) – Biomechanical analysis from video
- [RunStrong](runstrong.md) – Strength training tracking and planning
- [Coach G](coachg.md) – LLM-powered training advice
- [RunningHub Core](runninghub.md) – Fitness metrics, visualizations, dashboards

## 🧪 Demo
![runninghub_demo](images/RunningHub_Demo.gif)

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
