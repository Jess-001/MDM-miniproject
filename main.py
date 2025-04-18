from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 🔹 Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# 🔹 FastAPI app initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this if needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Load model and vectorizer
model = joblib.load("final_model_v2.sav")         # Ensure this file is in your backend directory
vectorizer = joblib.load("vectorizer_v2.pkl")     # Same here

# 🔹 Request body schema
class NewsRequest(BaseModel):
    content: str

# 🔹 Text preprocessing (same as used during training)
def preprocess_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special chars and digits
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered)

# 🔹 Prediction route
@app.post("/predict")
async def predict(request: NewsRequest):
    raw_text = request.content
    clean_text = preprocess_text(raw_text)
    vectorized_input = vectorizer.transform([clean_text])
    prediction = model.predict(vectorized_input)[0]
    return {"result": "Fake" if prediction == 1 else "Real"}
#uvicorn main:app --reload -for starting server
