

# RunStrong Strength Training

A modular, open-source project for runners and coaches to analyze running data, visualize training history, detect metric changes, and generate summaries using machine learning and language models.

## 🚀 Project Goals

* Provide a user-friendly web interface to evaluate and plan strength workouts

## 🔄 Data Sources

* **User Provided** 

## Database ERD

+---------------------+           +-----------------------+           +---------------------+
|   workout_routines  |           |   routine_exercises   |           |      exercises      |
+---------------------+           +-----------------------+           +---------------------+
| id (PK)             |<---------+| routine_id (FK)       |          +| id (PK)             |
| name                |           | exercise_id (FK)      +---------->| name                |
| date_created        |           | sets                  |           | description         |
+---------------------+           | reps                  |           | instructions        |
                                  | load_lbs              |           | exercise_type       |
                                  | order_index           |           | movement_pattern    |
                                  | notes                 |           | primary_muscles     |
                                  +-----------------------+           | secondary_muscles   |
                                                                      | muscle_groups       |
                                                                      | unilateral          |
                                                                      | difficulty_rating   |
                                                                      | prerequisites       |
                                                                      | progressions        |
                                                                      | regressions         |
                                                                      | equipment_required  |
                                                                      | equipment_optional  |
                                                                      | setup_time          |
                                                                      | space_required      |
                                                                      | rep_range_min       |
                                                                      | rep_range_max       |
                                                                      | tempo               |
                                                                      | range_of_motion     |
                                                                      | compound_vs_isolation|
                                                                      | injury_risk_level   |
                                                                      | contraindications   |
                                                                      | common_mistakes     |
                                                                      | safety_notes        |
                                                                      | image_url           |
                                                                      | video_url           |
                                                                      | gif_url             |
                                                                      | diagram_url         |
                                                                      | category            |
                                                                      | training_style      |
                                                                      | experience_level    |
                                                                      | goals               |
                                                                      | duration_minutes    |
                                                                      | popularity_score    |
                                                                      | alternatives        |
                                                                      | supersets_well_with |
                                                                      +---------------------+

Legend:
(PK) = Primary Key
(FK) = Foreign Key

Relationships:
- workout_routines.id ↔ routine_exercises.routine_id
- exercises.id ↔ routine_exercises.exercise_id


## 📦 Initial Features (Phase 1)

* [ ] 


## 📅 Planned Features (Backlog)

* [ ] Authentication (Auth0 or internal)
* [ ] REST API endpoints for mobile app access
* [ ] Integration with training plans & schedule data
* [ ] Personalized recommendations engine 
* [ ] Multi-user support and role-based access
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
