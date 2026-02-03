import streamlit as st
from agent.agent_logic import agent_decision

st.set_page_config(page_title="AI Disease & Emergency Agent", layout="centered")

st.title("🧠 Intelligent Disease & Emergency Analysis Agent")
st.write("Enter your symptoms below:")

symptoms = st.text_area("Example: fever, headache, vomiting")

if st.button("Analyze Patient"):
    if symptoms.strip() == "":
        st.warning("Please enter symptoms")
    else:
        disease, severity, emergency, treatment, desc, cause = agent_decision(symptoms)

        st.success("Analysis Complete")
        st.write("🦠 **Predicted Disease:**", disease)
        st.write("⚠ **Severity Level:**", severity)
        st.write("🚨 **Emergency:**", emergency)
        st.write("💊 **Treatment Type:**", treatment)
        st.write("📝 **Description:**", desc)
        st.write("🔍 **Possible Causes:**", cause)
