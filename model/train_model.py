import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("data/dataset.csv")
df = df.fillna("")

# Assume Symptom is text, weight is target
X_text = df['Symptom']
y = df['weight']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "disease_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully")