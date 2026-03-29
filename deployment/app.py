import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="debasishdas1985/tourism-package-predictor-model", filename="best_tourism_model_v1.joblib")

# Load the model
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Purchase Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for Visit with Us company that predicts whether customers are likely to purchase the Wellness Tourism Package based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to purchase the package.")

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Demographics")
    Age = st.number_input("Age (customer's age in years)", min_value=18, max_value=100, value=35)
    CityTier = st.selectbox("City Tier (development level)", [1, 2, 3])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Occupation = st.text_input("Occupation (e.g., Salaried, Freelancer, Self Employed)", value="Salaried")
    Designation = st.text_input("Designation (job title)", value="Manager")

with col2:
    st.subheader("Trip Details")
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    NumberOfTrips = st.number_input("Average Number of Trips Annually", min_value=0, max_value=20, value=3)
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [1, 2, 3, 4, 5])
    NumberOfChildrenVisiting = st.number_input("Number of Children (below age 5)", min_value=0, max_value=5, value=0)
    Passport = st.selectbox("Has Passport?", ["Yes", "No"])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

col3, col4 = st.columns(2)

with col3:
    st.subheader("Financial & Contact Info")
    MonthlyIncome = st.number_input("Monthly Income (in local currency)", min_value=0.0, value=50000.0)
    TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
    ProductPitched = st.text_input("Product Pitched", value="Wellness Package")

with col4:
    st.subheader("Pitch Details")
    PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score (1-10)", min_value=1, max_value=10, value=7)
    NumberOfFollowups = st.number_input("Number of Follow-ups by Salesperson", min_value=0, max_value=20, value=3)
    DurationOfPitch = st.number_input("Duration of Pitch (in minutes)", min_value=5, max_value=120, value=30)

# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': CityTier,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict Purchase Likelihood", use_container_width=True):
    try:
        prediction_proba = model.predict_proba(input_data)[0, 1]
        prediction = (prediction_proba >= classification_threshold).astype(int)

        # Display results with styling
        st.divider()
        if prediction == 1:
            st.success("Expected Outcome: Customer is LIKELY to PURCHASE the Wellness Tourism Package")
            st.metric("Purchase Confidence", f"{prediction_proba*100:.2f}%")
        else:
            st.warning("Expected Outcome: Customer is UNLIKELY to PURCHASE the Wellness Tourism Package")
            st.metric("Purchase Confidence", f"{prediction_proba*100:.2f}%")
        st.divider()
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please ensure all required fields are filled correctly.")

# Footer
st.markdown("---")
st.caption("Powered by Visit with Us MLOps Pipeline | Prediction Model v1.0 | Confidence Threshold: 45%")
