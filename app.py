import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Student Predictor", layout="centered")

# 🎨 Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    color: white;
}
.card {
    background-color: #1e293b;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    max-width: 520px;
    margin: auto;
}
.title {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 20px;
}
.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    color: white;
    border-radius: 10px;
    padding: 10px;
    border: none;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# 🧩 Card UI
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="title">🎓 Student Exam Score Predictor</div>', unsafe_allow_html=True)

# -------------------------------
# 📚 Study Hours (Slider)
# -------------------------------
st.subheader("📚 Study Hours")
study_hours = st.slider("Select Study Hours", min_value=0, max_value=10, step=1, value=2)
st.write(f"Selected: {study_hours} hrs")

# -------------------------------
# 📊 Attendance (Dropdown 0-100 step 10)
# -------------------------------
attendance = st.selectbox(
    "📊 Attendance (%)",
    options=list(range(0, 101, 10))
)

# -------------------------------
# 📝 Internal + Final Marks
# -------------------------------
internal_marks = st.selectbox("📝 Internal Marks", list(range(0, 101)))
final_marks = st.selectbox("📄 Final Exam Marks", list(range(0, 101)))

# -------------------------------
# 📚 Assignment Section
# -------------------------------
st.subheader("📚 Assignments")
total_assignments = st.selectbox("Total Assignments", list(range(1, 21)))

completed_count = 0
st.write("Mark completed assignments:")
cols = st.columns(5)
for i in range(total_assignments):
    with cols[i % 5]:
        if st.checkbox(f"A{i+1}", key=f"a{i}"):
            completed_count += 1
st.write(f"✅ Completed: {completed_count} / {total_assignments}")

# -------------------------------
# 💼 Part-time Job (for suggestions only)
# -------------------------------
part_time_job = st.selectbox("💼 Part time Job", ["No", "Yes"])

# -------------------------------
# 🔮 Prediction
# -------------------------------
if st.button("Predict Exam Score"):

    assignment_ratio = completed_count / total_assignments

    # Model expects 5 features: study_hours, attendance, internal_marks, final_marks, assignment_ratio
    input_data = np.array([
        [study_hours, attendance, internal_marks, final_marks, assignment_ratio]
    ])

    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))

    # 🎯 Result
    st.success(f"📊 Predicted Score: {prediction:.2f}")

    # 🎓 Grade
    if prediction >= 80:
        st.success("🎯 Grade: A (Excellent)")
    elif prediction >= 60:
        st.warning("🎯 Grade: B (Good)")
    else:
        st.error("🎯 Grade: C (Needs Improvement)")

    # 💡 Suggestions
    st.subheader("💡 Suggestions")
    if study_hours < 4:
        st.write("👉 Increase study hours")
    if attendance < 70:
        st.write("👉 Improve attendance")
    if assignment_ratio < 0.7:
        st.write("👉 Complete more assignments")
    if part_time_job == "Yes":
        st.write("👉 Balance job and study")

    # -------------------------------
    # 📊 Donut Chart (Attendance)
    # -------------------------------
    st.subheader("📊 Attendance Overview")

    present = attendance
    absent = 100 - attendance

    fig, ax = plt.subplots()
    ax.pie(
        [present, absent],
        labels=["Present", "Absent"],
        colors=["#10b981", "#ef4444"],
        autopct='%1.0f%%',
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='white'),
        textprops={'color':'white', 'fontsize':12}
    )

    ax.text(0, 0, f"{attendance}%", ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    ax.axis('equal')
    fig.patch.set_facecolor('#0f172a')  # Matches app background
    st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)