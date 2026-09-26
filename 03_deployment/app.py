import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(
    page_title="Classroom Attendance Predictor & Analytics",
    page_icon="🎓",
    layout="wide"
)

# Load pipeline model safely
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "best_attendance_model.pkl"),
        os.path.join(current_dir, "03_deployment", "best_attendance_model.pkl"),
        os.path.join(current_dir, "..", "03_deployment", "best_attendance_model.pkl"),
        "best_attendance_model.pkl"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("Model file not found.")

@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, "attendance_featured.csv"),
        os.path.join(current_dir, "01_datasets", "attendance_featured.csv"),
        os.path.join(current_dir, "..", "01_datasets", "attendance_featured.csv"),
        "attendance_featured.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

try:
    pipeline = load_model()
    df_featured = load_data()
    st.sidebar.success("✅ ML Pipeline & Dataset Loaded")
except Exception as e:
    st.sidebar.error("⚠️ Ensure model and dataset files are present in the directory.")
    st.stop()

# Tab Navigation
tab1, tab2 = st.tabs(["🚀 Real-Time Attendance Predictor", "📊 Exploratory Data Analytics (EDA)"])

# ----------------- TAB 1: PREDICTOR -----------------
with tab1:
    st.title("🎓 Classroom Attendance Predictive System")
    st.markdown("Forecast attendance percentages based on historical signals and timetable constraints.")

    st.sidebar.header("📋 Scheduling Parameters")
    subject = st.sidebar.selectbox(
        "Subject",
        ["Python Programming", "Database Management Systems", "Machine Learning", 
         "Operating Systems", "Web Development", "Software Engineering"]
    )
    day_of_week = st.sidebar.selectbox(
        "Day of the Week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    )
    lecture_number = st.sidebar.selectbox(
        "Lecture Slot",
        ["1st", "2nd", "3rd", "4th", "5th", "6th"]
    )
    weather = st.sidebar.selectbox(
        "Weather Condition",
        ["Sunny", "Cloudy", "Rainy"]
    )
    prev_attendance = st.sidebar.slider(
        "Previous Class Attendance (%)",
        30.0, 100.0, 75.0, 0.5
    )
    faculty_exp = st.sidebar.slider(
        "Faculty Experience (Years)",
        1, 25, 6
    )
    gap_days = st.sidebar.selectbox(
        "Gap Since Previous Class (Days)",
        [1, 2, 3, 4]
    )

    st.sidebar.subheader("📌 Contextual Signals")
    is_test_week = st.sidebar.checkbox("Internal Examination / Test Week")
    assignment_due = st.sidebar.checkbox("Assignment Deadline Today")
    holiday_adjacent = st.sidebar.checkbox("Holiday Adjacent (Day Before/After Break)")
    special_event = st.sidebar.checkbox("Campus Event / Festival Scheduled")

    slot_type = "Morning" if lecture_number in ["1st", "2nd", "3rd"] else "Afternoon"
    is_practical = 1.0 if subject in ["Python Programming", "Web Development"] else 0.0

    input_df = pd.DataFrame([{
        'Subject': subject,
        'Day of Week': day_of_week,
        'Lecture Number': lecture_number,
        'Slot_Type': slot_type,
        'Weather': weather,
        'Previous Lecture Attendance': float(prev_attendance),
        'Internal Test Week': 1.0 if is_test_week else 0.0,
        'Assignment Due': 1.0 if assignment_due else 0.0,
        'Holiday Before/After': 1.0 if holiday_adjacent else 0.0,
        'Special Event': 1.0 if special_event else 0.0,
        'Faculty Experience': float(faculty_exp),
        'Is_Practical': is_practical,
        'Gap_Days': float(gap_days)
    }])

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("⚙️ Configured Session Overview")
        st.table(pd.DataFrame({
            "Parameter": ["Subject", "Day", "Lecture Slot", "Class Format", "Timing Window", "Weather"],
            "Value": [subject, day_of_week, lecture_number, "Practical (Lab)" if is_practical == 1.0 else "Theory", slot_type, weather]
        }))

    with col2:
        st.subheader("📊 Forecasted Attendance")
        if st.button("🚀 Forecast Attendance", use_container_width=True, type="primary"):
            pred = max(0.0, min(100.0, pipeline.predict(input_df)[0]))
            total_enrolled = 60
            headcount = int(round((pred / 100.0) * total_enrolled))
            
            m1, m2 = st.columns(2)
            m1.metric("Predicted Attendance", f"{pred:.2f}%")
            m2.metric("Expected Headcount", f"{headcount} / {total_enrolled}")
            
            st.write("")
            if pred >= 75.0:
                st.success("🟢 **High Attendance Expected (≥ 75%)** — Routine schedule recommended.")
            elif pred >= 50.0:
                st.warning("🟡 **Moderate Attendance Expected (50% - 75%)** — Monitor slot scheduling.")
            else:
                st.error("🔴 **Low Attendance Warning (< 50%)** — Proactive timetable adjustment recommended.")

# ----------------- TAB 2: DATA VISUALIZATIONS -----------------
with tab2:
    st.title("📊 Attendance Trend & Factor Analysis")
    st.markdown("Visual exploration of attendance drivers across subjects, time slots, and academic calendar events.")
    
    if df_featured is not None:
        sns.set_theme(style="whitegrid")
        
        # Row 1
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Attendance Distribution by Subject")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_featured, x='Subject', y='Attendance Percentage', color='skyblue', ax=ax1)
            ax1.tick_params(axis='x', rotation=30)
            st.pyplot(fig1)
            
        with c2:
            st.markdown("#### Attendance by Lecture Slot")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=df_featured, x='Lecture Number', y='Attendance Percentage', hue='Slot_Type', palette='muted', ax=ax2)
            st.pyplot(fig2)

        st.divider()

        # Row 2
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Impact of Exams & Holiday Proximity")
            plot_df = df_featured.copy()
            plot_df['Test_Week_Label'] = plot_df['Internal Test Week'].map({1.0: 'Exam Week', 0.0: 'Regular Week'})
            plot_df['Holiday_Label'] = plot_df['Holiday Before/After'].map({1.0: 'Near Holiday', 0.0: 'Normal Day'})
            
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plot_df, x='Test_Week_Label', y='Attendance Percentage', hue='Holiday_Label', palette='coolwarm', ax=ax3)
            ax3.set_xlabel('')
            st.pyplot(fig3)

        with c4:
            st.markdown("#### Feature Correlation Heatmap")
            numeric_cols = [
                'Previous Lecture Attendance', 'Internal Test Week', 'Assignment Due',
                'Holiday Before/After', 'Special Event', 'Faculty Experience',
                'Is_Practical', 'Gap_Days', 'Attendance Percentage'
            ]
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            sns.heatmap(df_featured[numeric_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax4)
            ax4.tick_params(axis='x', rotation=45)
            st.pyplot(fig4)
    else:
        st.warning("⚠️ `attendance_featured.csv` is needed in the folder to render analytics charts.")