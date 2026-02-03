import joblib
import numpy as np
import os
import pandas as pd

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and vectorizer from the project root
model = joblib.load(os.path.join(script_dir, "../disease_model.pkl"))
vectorizer = joblib.load(os.path.join(script_dir, "../vectorizer.pkl"))

# Load the dataset for descriptions and causes
df = pd.read_csv(os.path.join(script_dir, "../data/dataset.csv"))

def predict_disease(symptoms):
    """
    Predict disease based on symptoms using simple keyword matching.
    """
    symptoms_lower = [s.lower() for s in symptoms]
    
    if any(s in symptoms_lower for s in ["fever", "cough", "sore throat"]):
        return "Respiratory Infection"
    elif any(s in symptoms_lower for s in ["headache", "nausea", "dizziness"]):
        return "Migraine or Neurological Issue"
    elif any(s in symptoms_lower for s in ["itching", "skin rash", "redness"]):
        return "Skin Condition"
    elif any(s in symptoms_lower for s in ["stomach pain", "vomiting", "diarrhea"]):
        return "Gastrointestinal Disorder"
    elif any(s in symptoms_lower for s in ["joint pain", "muscle pain", "fatigue"]):
        return "Musculoskeletal Disorder"
    elif any(s in symptoms_lower for s in ["chest pain", "shortness of breath"]):
        return "Cardiovascular Issue"
    else:
        return "General Illness"

def agent_decision(symptoms):
    """
    Analyze symptoms and return disease prediction, severity, emergency status, and treatment.
    """
    # Split symptoms into list
    symptom_list = [s.strip() for s in symptoms.split(',') if s.strip()]
    
    if not symptom_list:
        return "No symptoms provided", 0, "No", "None", "No description", "No cause"
    
    # Check if all symptoms are in the dataset
    for s in symptom_list:
        if not df[df['Symptom'].str.lower() == s.lower()].empty:
            continue
        else:
            return f"There is no data about this symptom: {s}", 0, "No", "None", "No description", "No cause"
    
    # Vectorize each symptom
    X_input = vectorizer.transform(symptom_list)
    
    # Predict weights (severities)
    predictions = model.predict(X_input)
    
    # Average severity
    avg_severity = np.mean(predictions)
    
    # Determine disease based on symptoms
    disease = predict_disease(symptom_list)
    
    # Severity level
    if avg_severity < 3:
        severity = "Low"
    elif avg_severity < 5:
        severity = "Medium"
    else:
        severity = "High"
    
    # Emergency
    emergency = "Yes" if avg_severity >= 5 else "No"
    
    # Treatment
    if severity == "Low":
        treatment = "Rest and monitor"
    elif severity == "Medium":
        treatment = "Consult doctor"
    else:
        treatment = "Seek immediate medical attention"
    
    # Get description and cause
    if len(symptom_list) == 1:
        symptom = symptom_list[0].lower()
        row = df[df['Symptom'].str.lower() == symptom]
        if not row.empty:
            desc = row['description'].iloc[0]
            cause = row['cause'].iloc[0]
        else:
            desc = "Description not available for this symptom"
            cause = "Cause not available for this symptom"
    else:
        # For multiple symptoms, generate combined description and cause
        descs = []
        causes = []
        for s in symptom_list:
            row = df[df['Symptom'].str.lower() == s.lower()]
            if not row.empty:
                descs.append(row['description'].iloc[0])
                causes.append(row['cause'].iloc[0])
        
        if descs:
            # Combine descriptions into one liner
            desc = f"Combination of symptoms: {'; '.join(descs[:2])}" + ("..." if len(descs) > 2 else "")
        else:
            desc = "Descriptions not available"
        
        if causes:
            # Combine causes into one liner, removing duplicates
            unique_causes = list(set(causes))
            cause = f"Possible causes: {'; '.join(unique_causes[:2])}" + ("..." if len(unique_causes) > 2 else "")
        else:
            cause = "Causes not available"
    
    return disease, f"{severity} ({avg_severity:.1f})", emergency, treatment, desc, cause