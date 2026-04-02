
import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Fake Resume Detector", page_icon="🧾")


st.title("🧾 Fake Resume Detection System")
st.markdown("### 🔍 Check whether a resume is **Fake or Genuine**")

st.write("Enter candidate details below:")


col1, col2 = st.columns(2)

with col1:
    experience = st.number_input("👨‍💼 Years of Experience", 0, 40)
    education = st.selectbox("🎓 Education Level", 
                             ["UG", "PG", "PhD"])

with col2:
    skills = st.number_input("🛠️ Number of Skills", 0, 50)
    projects = st.number_input("📁 Number of Projects", 0, 20)

certifications = st.number_input("📜 Certifications Count", 0, 20)

# Convert education to numeric
if education == "UG":
    education_val = 0
elif education == "PG":
    education_val = 1
else:
    education_val = 2

# -------------------------------
# Step 6: Predict Button
# -------------------------------
if st.button("🚀 Predict"):

    # Step 7: Prepare input data
    input_data = np.array([[experience, education_val, skills, projects, certifications]])

    # Step 8: Scale input
    input_data = scaler.transform(input_data)

    # Step 9: Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    # -------------------------------
    # Step 10: Show Result
    # -------------------------------
    st.subheader("🔎 Result:")

    if prediction[0] == 1:
        st.error("⚠️ Fake Resume Detected")
    else:
        st.success("✅ Genuine Resume")

    # Show confidence
    st.write(f"Confidence Score: {round(max(probability[0])*100, 2)}%")

# -------------------------------
# Step 11: Footer
# -------------------------------
st.markdown("---")
st.caption("🎓 Project: Fake Resume Detection using Machine Learning")
st.caption("👩‍💻 Developed by B.Tech CSE Student")