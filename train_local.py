import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

# 1. Load the cleaned and engineered dataset
df = pd.read_csv("attendance_featured.csv")

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

# 6. Save model locally
joblib.dump(final_pipeline, "best_attendance_model.pkl")
print("✅ Model retrained and saved locally with matching scikit-learn version!")