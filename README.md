# VisionaryX-AI
An intelligent web-based healthcare agent that analyzes patient symptoms to predict possible diseases, assess severity and emergency level, and explain symptom causes using machine learning and AI.

🧠 Intelligent Disease & Emergency Analysis Agent

An agentic AI-based healthcare decision support system that analyzes patient symptoms to predict possible diseases, assess severity and emergency level, and generate human-readable explanations of symptom descriptions and causes using machine learning and knowledge-based reasoning through a web interface.

📌 Project Overview

Healthcare systems often struggle with efficient and fast patient assessment due to high patient volume and limited medical staff. This project aims to provide an intelligent, web-based healthcare assistant that performs preliminary symptom analysis and helps identify potential diseases, urgency level, and treatment category.

The system accepts one or more symptoms from the user and performs the following tasks:
• Predicts the most likely disease
• Determines the severity level (Low / Medium / High)
• Identifies whether the condition is an emergency
• Suggests the type of treatment required
• Explains symptom meaning and cause in simple language

For a single symptom, the system provides a detailed description and cause.
For multiple symptoms, it generates a concise, combined explanation using AI-based logic.

This project demonstrates how machine learning and rule-based reasoning can be combined to build an agentic AI system for healthcare assistance.

🎯 Objectives

• To design an AI system that predicts diseases based on symptoms
• To analyze severity and emergency conditions
• To generate explanations of symptoms and causes
• To provide a user-friendly web interface
• To demonstrate agentic AI behavior in healthcare
• To support decision-making for patients and healthcare providers

⚙️ Key Functionalities
1. Disease Prediction

Predicts the most probable disease based on user-entered symptoms using a trained machine learning model.

2. Severity Assessment

Classifies the condition into Low, Medium, or High severity using medical mapping rules.

3. Emergency Detection

Identifies whether immediate medical attention is required based on severity and disease type.

4. Treatment Recommendation

Suggests appropriate treatment category such as home care, doctor consultation, or hospital care.

5. Symptom Description Generator

For a single symptom, displays its meaning in simple terms.

6. Symptom Cause Analyzer

Explains the possible reason behind the symptom.

7. Multi-Symptom Reasoning

For multiple symptoms, produces a short combined description and cause using intelligent logic.

8. Web-Based User Interface

Provides an interactive browser-based interface for easy testing and demonstration.

🏗️ System Architecture
User Input (Symptoms)
        ↓
Web Interface (Streamlit)
        ↓
Symptom Processing Module
        ↓
Machine Learning Model
        ↓
Decision & Rule Engine
        ↓
Explanation Generator
        ↓
Final Output (Disease, Severity, Emergency, Treatment, Description, Cause)

📁 Project Structure
Disease_Agent_Project/
│
├── data/
│   └── dataset.csv              # Main training dataset
│
├── model/
│   ├── train_model.py           # Model training script
│   
├── agent/
│   └── agent_logic.py           # Core decision-making logic
│
├── app.py                       # Web application (frontend)
├── requirements.txt             # Required libraries
├── disease_model.pkl            # Trained model
├── symptom_columns.pkl          # Symptom feature list
└── README.md                    # Project documentation

📊 Dataset

The model is trained using a symptom–disease dataset obtained from Kaggle.
The dataset contains binary symptom indicators mapped to disease labels.

Additionally, a custom knowledge dataset (symptom_info.csv) is used to store:
• Symptom descriptions
• Causes of symptoms

This allows the system to explain symptoms in a human-readable form.

🧪 Technologies Used

• Python
• Pandas
• NumPy
• Scikit-learn
• Streamlit
• Joblib

🚀 Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd Disease_Agent_Project

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Train the Model
python model/train_model.py

5. Run the Web Application
streamlit run app.py


Open browser at:

http://localhost:8501

🧠 How It Works

User enters one or more symptoms

Symptoms are converted into a machine-readable format

ML model predicts the disease

Severity and emergency level are calculated

Symptom meaning and cause are generated

Results are displayed in the browser

🩺 Example Use Case

Input:

high_fever, headache, vomiting


Output:
🦠 Predicted Disease: Migraine or Neurological Issue

⚠ Severity Level: Medium (4.7)

🚨 Emergency: No

💊 Treatment Type: Consult doctor

📝 Description: Combination of symptoms: Elevated body temperature above 101°F; Pain in the head or neck area...

🔍 Possible Causes: Possible causes: Commonly caused by infections, food poisoning, pregnancy, or gastrointestinal issues; Often caused by tension, migraines, dehydration, or infections...

🔐 Ethical Considerations

• This system does not replace a doctor
• It provides preliminary assistance only
• User data should not be stored permanently
• Predictions must be used for educational purposes

🔮 Future Enhancements

• Voice-based symptom input
• Mobile application integration
• Doctor dashboard
• Confidence score display
• PDF patient report generation
• Integration with wearable devices
• Use of large language models for explanations

🏆 Significance

This project demonstrates the application of agentic AI in healthcare by combining machine learning with reasoning and explanation generation. It serves as an effective example of how AI can assist in early-stage medical assessment while remaining interpretable and user-friendly.

👨‍💻 Contributors

• Poorv Khatri
• Nikhil Jani
• Deven Jadav
• Yash Soni
