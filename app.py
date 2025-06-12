import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="SMS Spam Detector")
st.title("📩 SMS Spam Classifier")
st.write("Enter an SMS message to check if it's spam or not.")

# Input box
user_input = st.text_area("Your message:")

if st.button("Classify"):
    if user_input:
        # Transform input
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        if prediction == "spam":
            st.error("🚫 Spam Detected!")
        else:
            st.success("✅ Message is Not Spam.")
    else:
        st.warning("Please enter a message.")




# import streamlit as st
# import joblib

# # Load model and vectorizer
# model = joblib.load("spam_classifier_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Streamlit UI
# st.set_page_config(page_title="SMS Spam Detector")
# st.title("📩 SMS Spam Classifier")
# st.write("Enter an SMS message to check if it's spam or not.")

# # Input box
# user_input = st.text_area("Your message:")

# if st.button("Classify"):
#     if user_input:
#         # Transform input
#         transformed_input = vectorizer.transform([user_input])
#         prediction = model.predict(transformed_input)[0]

#         if prediction == "spam":
#             st.error("🚫 Spam Detected!")
#         else:
#             st.success("✅ Message is Not Spam.")
#     else:
#         st.warning("Please enter a message.")