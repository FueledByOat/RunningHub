# Running Analytics Hub

A modular, open-source project for runners and coaches to analyze running data, visualize training history, detect metric changes, and generate summaries using machine learning and language models.

## 🚀 Project Goals

* Provide a user-friendly web interface to visualize and analyze running data.
* Offer detailed breakdowns of performance metrics over time.
* Detect changes in performance using machine learning.
* Summarize recent workouts and trends using a local LLM.
* Use open-source tools, containerized deployment, and modular architecture.

## 🧱 Technology Stack

| Feature           | Tool(s)                                                |
| ----------------- | ------------------------------------------------------ |
| Web Interface     | Flask                                                  |
| Dashboards        | Dash Leaflet                                           |
| Data Storage      | SqLite + PostGIS                                   |
| GPX Parsing       | `gpxpy`, `pandas`, `shapely`                           |
| Metrics Analysis  | `pandas`, `scipy`, `plotly`, `numpy`                   |
| ML Models         | `scikit-learn`, `ruptures`, `xgboost`                  |
| LLM Summaries     | `transformers`, `ollama`, `langchain`                  |
| Data Ingestion    | Custom scripts, Prefect/Airflow (optional)             |
| Authentication    | Auth0, Flask-Login (optional)                          |
| Containerization  | Docker, Docker Compose                                 |
| DevOps Monitoring | Grafana + Prometheus (for server/app health)           |

## 🔄 Data Sources

* **Strava API** (preferred for historical + social data)
* **Coros Export Files** (local device-based JSON/CSV/GPX)
* **Apple Health** (via HealthKit exports or 3rd-party tools)

## 📦 Initial Features (Phase 1)

* [ ] Dockerized Flask backend + Postgres DB
* [ ] GPX file upload and heatmap visualization
* [ ] Metrics analysis dashboard (HR, cadence, stride length, power)
* [ ] Time-based filters for session comparison
* [ ] Simple ML change detection (e.g., stride length anomalies)
* [ ] LLM summarizer API for weekly insights

## 🛠️ Architecture (Simplified)

```
[ User Interface (Flask/Dash) ]
          |
  [ Metrics API / Upload API ]
          |
    [ Postgres + PostGIS ]
          |
   [ ML Pipeline (sklearn, etc) ]
          |
 [ Local LLM Summary Engine ]
          |
 [ Docker Compose / Monitoring Layer ]
```

## 📅 Planned Features (Backlog)

* [ ] Authentication (Auth0 or internal)
* [ ] Prefect-based data ingestion workflows
* [ ] REST API endpoints for mobile app access
* [ ] Integration with training plans & schedule data
* [ ] Personalized recommendations engine (e.g., "your stride length has dropped 4% since your last race")
* [ ] Multi-user support and role-based access
* [ ] Export to PDF reports
* [ ] Real-time metrics overlay via smartwatch connection (stretch goal)

## 📚 Documentation

* Written with Markdown + MkDocs for clean, static documentation.
* Dev and data onboarding guides.
* Example GPX, JSON, and dashboard previews included.

## 🤝 Contribution Guidelines

* Fork the repo and set up Docker locally.
* Submit well-documented pull requests.
* Open issues for bugs, feature requests, or ideas.

## 📜 License

MIT License

---

Feel free to contribute or adapt this hub for your own running or coaching needs!
