import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
import json
import os
import re

warnings.filterwarnings("ignore")

# load trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Student Predictor", layout="centered")

# prediction history file
HISTORY_FILE = "history.json"

# create history file if not exists
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w") as f:
        json.dump({}, f)

# load history
def load_history():
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

# save history
def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# user database file
USER_FILE = "users.json"

# create user file if not exists
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({}, f)

# load users
def load_users():
    with open(USER_FILE, "r") as f:
        return json.load(f)

# save users
def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# login signup page
if not st.session_state.logged_in:

    st.title("🔐 Login / Signup")

    option = st.selectbox("Choose Option", ["Login", "Signup"])

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # email validation
    def validate_email(email):
        return re.match(r"^[a-zA-Z0-9._%+-]+@gmail\.com$", email)

    # password validation
    def validate_password(password):
        if len(password) < 8:
            return False
        if not re.search(r"[A-Z]", password):
            return False
        if not re.search(r"[0-9]", password):
            return False
        if not re.search(r"[!@#$%^&*]", password):
            return False
        return True

    users = load_users()

    if st.button("Submit"):

        if not validate_email(email):
            st.error("Enter valid gmail address")

        elif not validate_password(password):
            st.error("Password must be 8 characters with uppercase, number and symbol")

        else:

            # signup logic
            if option == "Signup":

                if email in users:
                    st.error("Email already registered. Please login.")
                else:
                    users[email] = password
                    save_users(users)

                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.success("Signup successful")
                    st.rerun()

            # login logic
            elif option == "Login":

                if email not in users:
                    st.error("Email not found")
                elif users[email] != password:
                    st.error("Incorrect password")
                else:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.success("Login successful")
                    st.rerun()

# main application
else:

    st.title("🎓 Student Exam Score Predictor")

    # logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.subheader("Enter Student Details")

    # input features
    study_hours = st.slider("Study Hours Per Day", 0, 12, 3)
    social_media = st.slider("Social Media Hours", 0, 10, 2)
    attendance = st.slider("Attendance Percentage", 0, 100, 75)
    sleep_hours = st.slider("Sleep Hours", 4, 10, 7)
    part_time_job = st.selectbox("Part Time Job", ["No", "Yes"])
    extracurricular = st.selectbox("Extracurricular Participation", ["No", "Yes"])
    internal_marks = st.slider("Internal Marks", 0, 100, 60)
    assignment_score = st.slider("Assignment Score", 0, 100, 70)

    # prediction button
    if st.button("Predict Exam Score"):

        # encode categorical values
        ptj_encoded = 1 if part_time_job == "Yes" else 0
        extra_encoded = 1 if extracurricular == "Yes" else 0

        # model input
        input_data = np.array([[study_hours,
                                social_media,
                                ptj_encoded,
                                attendance,
                                sleep_hours,
                                extra_encoded,
                                internal_marks,
                                assignment_score]])

        # predict
        prediction = model.predict(input_data)[0]

        # limit score
        prediction = max(0, min(100, prediction))

        st.success(f"Predicted Exam Score: {prediction:.2f}")

        # grade
        if prediction >= 80:
            st.success("Grade: A (Excellent)")
        elif prediction >= 60:
            st.warning("Grade: B (Good)")
        else:
            st.error("Grade: C (Needs Improvement)")

        # suggestions
        st.subheader("Suggestions")

        if study_hours < 4:
            st.write("Increase study hours")

        if social_media > 5:
            st.write("Reduce social media usage")

        if attendance < 70:
            st.write("Improve attendance")

        if sleep_hours < 6:
            st.write("Get proper sleep")

        if part_time_job == "Yes":
            st.write("Balance job and study")

        # combined student feature chart
        st.subheader("Student Habit and Academic Analysis")

        features = [
            "Study Hours",
            "Social Media",
            "Attendance",
            "Sleep Hours",
            "Internal Marks",
            "Assignment Score"
        ]

        values = [
            study_hours,
            social_media,
            attendance,
            sleep_hours,
            internal_marks,
            assignment_score
        ]

        fig, ax = plt.subplots()

        # thinner bars
        ax.bar(features, values, width=0.4)

        ax.set_ylabel("Values")
        ax.set_title("Student Habits and Academic Metrics")

        plt.xticks(rotation=30)

        st.pyplot(fig)

        # attendance donut chart
        st.subheader("Attendance Overview")

        present = attendance
        absent = 100 - attendance

        fig2, ax2 = plt.subplots()

        ax2.pie(
            [present, absent],
            labels=["Present", "Absent"],
            autopct='%1.0f%%',
            startangle=90,
            wedgeprops=dict(width=0.4)
        )

        ax2.text(0, 0, f"{attendance}%", ha='center', va='center', fontsize=16)
        ax2.axis("equal")

        st.pyplot(fig2)

        # save prediction history
        history = load_history()
        user = st.session_state.user_email

        if user not in history:
            history[user] = []

        history[user].append({
            "study_hours": study_hours,
            "social_media": social_media,
            "attendance": attendance,
            "sleep_hours": sleep_hours,
            "internal_marks": internal_marks,
            "assignment_score": assignment_score,
            "score": float(prediction)
        })

        save_history(history)

        # performance history chart
        st.subheader("Performance History")

        user_history = history.get(user, [])

        scores = [h["score"] for h in user_history]
        attempts = list(range(1, len(scores) + 1))

        fig3, ax3 = plt.subplots()

        ax3.plot(attempts, scores, marker="o")

        ax3.set_xlabel("Prediction Attempt")
        ax3.set_ylabel("Predicted Score")
        ax3.set_title("Student Performance Over Time")

        st.pyplot(fig3)

        # improvement analysis
        if len(scores) >= 2:
            change = scores[-1] - scores[-2]

            if change > 0:
                st.success(f"Performance improved by {change:.2f} points since last prediction")
            elif change < 0:
                st.warning(f"Performance decreased by {abs(change):.2f} points since last prediction")
            else:
                st.info("Performance stayed the same as last prediction")