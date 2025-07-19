import streamlit as st
import joblib

# Load pipeline (includes vectorizer + classifier)
model = joblib.load("spam_model.pkl")

# App UI
st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 SMS Spam Detector")
st.write("Enter a message and find out whether it's **Spam** or **Ham** (not spam).")

# Input
message = st.text_area("✉️ Your message:", height=150)

if st.button("Predict"):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        prediction = model.predict([message])[0]
        proba = model.predict_proba([message])[0]

        if prediction == 1:
            st.error(f"🚫 Spam Detected! ({proba[1]*100:.2f}% confidence)")
        else:
            st.success(f"✅ This is a Ham message! ({proba[0]*100:.2f}% confidence)")
