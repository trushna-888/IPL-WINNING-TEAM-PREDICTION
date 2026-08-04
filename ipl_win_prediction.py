# ============================================================
# IPL WINNING TEAM PREDICTION USING MACHINE LEARNING
# PART 1 : IMPORT LIBRARIES & LOAD DATA
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("=" * 60)
print("        IPL WINNING TEAM PREDICTION")
print("=" * 60)

# ------------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------------

print("\nLoading datasets...")

matches = pd.read_csv("matches.csv")
deliveries = pd.read_csv("deliveries.csv")

print("Datasets Loaded Successfully!")

# ------------------------------------------------------------
# DISPLAY BASIC INFORMATION
# ------------------------------------------------------------

print("\nMatches Dataset Shape :", matches.shape)
print("Deliveries Dataset Shape :", deliveries.shape)

print("\nMatches Columns")
print(matches.columns.tolist())

print("\nDeliveries Columns")
print(deliveries.columns.tolist())

# ------------------------------------------------------------
# REMOVE MATCHES WITHOUT WINNER
# ------------------------------------------------------------

matches = matches.dropna(subset=["winner"])

print("\nAfter Removing No Result Matches")
print("Matches Shape :", matches.shape)

# ------------------------------------------------------------
# MERGE DATASETS
# ------------------------------------------------------------

data = deliveries.merge(
    matches,
    left_on="match_id",
    right_on="id"
)

print("\nMerged Dataset Shape :", data.shape)

print("\nFirst Five Rows")
print(data.head())

print("\nPart 1 Completed Successfully!")
# ============================================================
# PART 2 : DATA PREPROCESSING & FEATURE ENGINEERING
# ============================================================

print("\nPreparing Match Data...")

# Use only second innings
data = data[data["inning"] == 2].copy()

# Current Score
data["current_score"] = data.groupby("match_id")["total_runs"].cumsum()

# Wickets Lost
data["wickets"] = data.groupby("match_id")["is_wicket"].cumsum()

# Balls Bowled
data["balls_bowled"] = data["over"] * 6 + data["ball"]

# Balls Left
data["balls_left"] = 120 - data["balls_bowled"]

# Runs Left
data["runs_left"] = data["target_runs"] - data["current_score"]

# Remove invalid rows
data = data[
    (data["balls_left"] > 0) &
    (data["runs_left"] >= 0)
]

# Current Run Rate
data["crr"] = (
    data["current_score"] * 6
) / data["balls_bowled"].replace(0, 1)

# Required Run Rate
data["rrr"] = (
    data["runs_left"] * 6
) / data["balls_left"]

# Match Result
data["result"] = np.where(
    data["batting_team"] == data["winner"],
    1,
    0
)

# Keep required columns
final_df = data[
    [
        "batting_team",
        "bowling_team",
        "city",
        "runs_left",
        "balls_left",
        "wickets",
        "target_runs",
        "crr",
        "rrr",
        "result"
    ]
].copy()

# Remove missing values
final_df.dropna(inplace=True)

print("\nFeature Engineering Completed!")

print("\nFinal Dataset Shape :", final_df.shape)

print("\nSample Dataset")
print(final_df.head())
# ============================================================
# PART 3 : MODEL TRAINING
# ============================================================

print("\nTraining Machine Learning Model...")

# -------------------------------
# INPUT FEATURES
# -------------------------------

X = final_df.drop("result", axis=1)
y = final_df["result"]

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------
# ONE HOT ENCODING
# -------------------------------

trf = ColumnTransformer(
    transformers=[
        (
            "trf",
            OneHotEncoder(handle_unknown="ignore"),
            ["batting_team", "bowling_team", "city"]
        )
    ],
    remainder="passthrough"
)

# -------------------------------
# LOGISTIC REGRESSION MODEL
# -------------------------------

pipe = Pipeline(
    steps=[
        ("step1", trf),
        ("step2", LogisticRegression(max_iter=1000))
    ]
)

# -------------------------------
# TRAIN MODEL
# -------------------------------

pipe.fit(X_train, y_train)

print("Model Training Completed!")

# -------------------------------
# MODEL ACCURACY
# -------------------------------

prediction = pipe.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\nModel Accuracy : {:.2f}%".format(accuracy * 100))
# ============================================================

# PART 5 : PROFESSIONAL USER INTERFACE
# ============================================================

teams = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Royal Challengers Bangalore",
    "Kolkata Knight Riders",
    "Delhi Capitals",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Punjab Kings",
    "Lucknow Super Giants",
    "Gujarat Titans"
]

print("\n" + "="*60)
print("             IPL WIN PREDICTOR")
print("="*60)

print("\nAvailable Teams\n")

for i, team in enumerate(teams, start=1):
    print(f"{i}. {team}")

batting_team = teams[int(input("\nSelect Batting Team (1-10): ")) - 1]
bowling_team = teams[int(input("Select Bowling Team (1-10): ")) - 1]

city = input("\nEnter Match City : ")

target = int(input("Target Score : "))
current_score = int(input("Current Score : "))
overs = float(input("Overs Completed : "))
wickets = int(input("Wickets Lost : "))

balls_bowled = int(overs * 6)
balls_left = 120 - balls_bowled
runs_left = target - current_score

crr = current_score / overs if overs != 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left != 0 else 0

input_df = pd.DataFrame({
    "batting_team":[batting_team],
    "bowling_team":[bowling_team],
    "city":[city],
    "runs_left":[runs_left],
    "balls_left":[balls_left],
    "wickets":[wickets],
    "target_runs":[target],
    "crr":[crr],
    "rrr":[rrr]
})

result = pipe.predict_proba(input_df)

lose = result[0][0]
win = result[0][1]

print("\n" + "="*60)
print("           WINNING PROBABILITY")
print("="*60)

print(f"\n🏏 {batting_team:<30} : {win*100:.2f}%")
print(f"🏏 {bowling_team:<30} : {lose*100:.2f}%")

print("\n" + "="*60)

if win > lose:
    print("🏆 Predicted Winner :", batting_team)
else:
    print("🏆 Predicted Winner :", bowling_team)

print("="*60)
# ============================================================
# PART 6 : SAVE TRAINED MODEL
# ============================================================

import pickle

print("\nSaving Model...")

with open("ipl_win_predictor.pkl", "wb") as file:
    pickle.dump(pipe, file)

print("Model Saved Successfully!")

print("\n" + "="*60)
print("        PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print("\nFiles Created Successfully")
print("----------------------------")
print("✔ ipl_win_prediction.py")
print("✔ ipl_win_predictor.pkl")

print("\nThank You for Using IPL Winning Team Predictor")
print("="*60)