import joblib
import numpy as np

model = joblib.load("../model/disease_model.pkl")
symptom_columns = joblib.load("../model/symptom_columns.pkl")

severity_map = {
    "Dengue": "High",
    "Malaria": "High",
    "Flu": "Medium",
    "Common Cold": "Low",
    "Pneumonia": "High"
}

treatment_map = {
    "Dengue": "Hospital care and fluids",
    "Malaria": "Doctor consultation",
    "Flu": "Doctor consultation",
    "Common Cold": "Home care and rest",
    "Pneumonia": "Emergency hospital care"
}

def agent_decision(symptoms_text):
    user_symptoms = symptoms_text.lower().split()

    input_vector = np.zeros(len(symptom_columns))

    for i, symptom in enumerate(symptom_columns):
        if symptom.replace("_", " ") in symptoms_text.lower():
            input_vector[i] = 1

    input_vector = input_vector.reshape(1, -1)

    disease = model.predict(input_vector)[0]
    severity = severity_map.get(disease, "Medium")
    treatment = treatment_map.get(disease, "Doctor consultation")
    emergency = "Yes" if severity == "High" else "No"

    return disease, severity, emergency, treatment
