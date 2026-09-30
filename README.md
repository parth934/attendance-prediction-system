# Classroom Attendance Prediction & Analytics System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://attendance-predictor.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

A Machine Learning powered web application built with **Streamlit** and **Scikit-Learn (Gradient Boosting Regressor)** to forecast classroom attendance percentages based on timetable schedules, weather signals, previous attendance, exam proximity, and calendar events.

---

## Features

- **Real-Time Attendance Predictor:**
  - Interactive parameters: Subject, Day of the Week, Lecture Slot, Weather condition, Previous lecture attendance, Faculty experience, and Gap days.
  - Contextual toggles: Internal exam week, Assignment deadlines, Holiday proximity, and Campus events.
  - Generates forecasted attendance percentage and estimated classroom headcount out of 60 students.
  - Actionable scheduling advisory tags (High, Moderate, Low attendance recommendations).

- **Exploratory Data Analytics (EDA):**
  - Attendance distribution across subjects (Boxplot).
  - Attendance variation across lecture slots and timing windows (Morning vs. Afternoon).
  - Impact analysis of exam weeks and holiday proximity.
  - Multi-factor correlation heatmap.

---

## Repository Structure

```text
attendance_capstone/
├── 01_datasets/
│   ├── raw_classroom_attendance_500.csv    # Raw dataset (500 lecture sessions)
│   └── attendance_featured.csv             # Feature-engineered dataset
│
├── 02_source_code/
│   ├── train_local.py                      # Model training & preprocessing pipeline
│   └── requirements.txt                    # Project dependencies
│
├── 03_deployment/
│   ├── app.py                              # Streamlit web application
│   ├── best_attendance_model.pkl           # Pre-trained Gradient Boosting pipeline
│   ├── attendance_featured.csv             # Dataset for interactive analytics
│   └── requirements.txt                    # Deployment dependencies
│
├── 04_documentation/
│   └── attendance_capstone_report.docx     # Capstone academic report
│
├── requirements.txt                        # Root dependencies for cloud hosting
├── .gitignore                              # Git exclusion rules
└── README.md                               # Project documentation & setup guide
```

---

## Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/parth934/attendance-prediction-system.git
   cd attendance-prediction-system
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run 03_deployment/app.py
   ```

---

## Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended)
1. Fork or push this repository to GitHub (`parth934/attendance-prediction-system`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **Create app** > Select repo `parth934/attendance-prediction-system`.
4. Set **Main file path** to `03_deployment/app.py`.
5. Click **Deploy!**

### Option 2: Render
- **Environment:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `streamlit run 03_deployment/app.py --server.port $PORT --server.address 0.0.0.0`

---

## Author & Acknowledgements
- **Author:** Parth
- **Tech Stack:** Python, Streamlit, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
