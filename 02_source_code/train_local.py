import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

import os

# 1. Load the cleaned and engineered dataset
possible_dataset_paths = [
    "attendance_featured.csv",
    os.path.join("01_datasets", "attendance_featured.csv"),
    os.path.join("..", "01_datasets", "attendance_featured.csv"),
    os.path.join(os.path.dirname(__file__), "..", "01_datasets", "attendance_featured.csv"),
    os.path.join(os.path.dirname(__file__), "attendance_featured.csv")
]

dataset_path = None
for p in possible_dataset_paths:
    if os.path.exists(p):
        dataset_path = p
        break

if not dataset_path:
    raise FileNotFoundError("Could not locate attendance_featured.csv")

print(f"📂 Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)

# 2. Features and Target
X = df.drop(columns=['Attendance Percentage'])
y = df['Attendance Percentage']

categorical_cols = ['Subject', 'Day of Week', 'Lecture Number', 'Slot_Type', 'Weather']
numerical_cols = [
    'Previous Lecture Attendance',
    'Internal Test Week',
    'Assignment Due',
    'Holiday Before/After',
    'Special Event',
    'Faculty Experience',
    'Is_Practical',
    'Gap_Days'
]

# 3. Preprocessor Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# 4. Final Best Model Pipeline (Gradient Boosting)
final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, random_state=42))
])

# 5. Fit on entire cleaned data
final_pipeline.fit(X, y)

# 6. Save model locally and to deployment folder
save_paths = [
    "best_attendance_model.pkl",
    os.path.join("03_deployment", "best_attendance_model.pkl"),
    os.path.join(os.path.dirname(__file__), "..", "03_deployment", "best_attendance_model.pkl")
]

for sp in save_paths:
    try:
        dir_name = os.path.dirname(sp)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        joblib.dump(final_pipeline, sp)
    except Exception:
        pass

print("✅ Model retrained and saved to root and 03_deployment folders!")