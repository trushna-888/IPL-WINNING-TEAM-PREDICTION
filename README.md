# 🏏 IPL Winning Team Prediction using Machine Learning

## 📌 Project Overview

The IPL Winning Team Prediction project is a Machine Learning application that predicts the probability of a team winning an Indian Premier League (IPL) match based on the current match situation. The prediction is made using historical IPL match data and Machine Learning algorithms.

---

## 🎯 Objective

The objective of this project is to estimate the winning probability of both teams during a live IPL match by analyzing match statistics such as runs, wickets, overs, target score, batting team, bowling team, and venue.

---

## 📂 Dataset

The model is trained using historical IPL match data containing information such as:

- Match ID
- Batting Team
- Bowling Team
- Venue
- Target Score
- Current Score
- Overs Completed
- Wickets Lost
- Runs Left
- Balls Left
- Result (Winning Team)

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Pickle
- Jupyter Notebook

---

## 🤖 Machine Learning Algorithm

- Logistic Regression

---

## 📊 Input Features

- Batting Team
- Bowling Team
- Venue
- Target Score
- Current Score
- Overs Completed
- Wickets Lost
- Runs Left
- Balls Left
- Current Run Rate (CRR)
- Required Run Rate (RRR)

---

## 📈 Output

The model predicts:

- 🟢 Winning Probability of the Batting Team
- 🔴 Winning Probability of the Bowling Team

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/trushna-888/IPL-Winning-Team-Prediction.git
```

### 2. Navigate to the project folder

```bash
cd IPL-Winning-Team-Prediction
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
IPL-Winning-Team-Prediction/
│
├── app.py
├── pipe.pkl
├── matches.csv
├── deliveries.csv
├── requirements.txt
├── README.md
└── images/
```

---

## 🚀 Future Improvements

- Improve prediction accuracy using advanced ML models
- Support upcoming IPL seasons
- Add player performance statistics
- Display match analytics and visualizations
- Deploy the application on Streamlit Cloud

---


## 👩‍💻 Author

**Trushna Bankar**

GitHub: https://github.com/trushna-888

---

## ⭐ If you found this project useful, don't forget to Star the repository!
