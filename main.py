from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load your model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class NewsRequest(BaseModel):
    content: str

@app.post("/predict")
async def predict(request: NewsRequest):
    text = request.content
    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    return {"result": "Fake" if prediction == 1 else "Real"}
