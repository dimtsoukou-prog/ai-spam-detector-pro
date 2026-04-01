from fastapi import FastAPI
from pydantic import BaseModel
import joblib


model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = FastAPI(title="AI Spam Detector API")

class SMS(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "AI Model is Live", "accuracy": "98.39%"}

@app.post("/predict")
def predict_spam(sms: SMS):
    text_vec = vectorizer.transform([sms.text])
    
    prediction = model.predict(text_vec)[0]
    
    return {
        "message": sms.text,
        "is_spam": True if prediction == 'spam' else False,
        "classification": prediction
    }