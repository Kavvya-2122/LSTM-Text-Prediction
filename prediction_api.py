
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize app
app = FastAPI(
    title="LSTM Text Prediction API",
    description="Predicts the next words in a sequence using an LSTM model",
    version="1.0.0"
)

# Load model and tokenizer
model = load_model("lstm_text_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

SEQ_LENGTH = 10
VOCAB_SIZE = 5000

# Request body structure
class PredictRequest(BaseModel):
    seed_text: str
    num_words: int = 5

# Response body structure
class PredictResponse(BaseModel):
    input_text: str
    predicted_text: str
    next_words: list

# Root endpoint
@app.get("/")
def home():
    return {
        "message": "LSTM Text Prediction API is running!",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "health": "/health"
        }
    }

# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "model": "LSTM", "vocab_size": VOCAB_SIZE}

# Main prediction endpoint
@app.post("/predict")
def predict(request: PredictRequest):
    result = request.seed_text
    next_words = []

    for _ in range(request.num_words):
        # Tokenize input
        token_list = tokenizer.texts_to_sequences([result])[0]

        # Pad sequence
        token_list = pad_sequences(
            [token_list],
            maxlen=SEQ_LENGTH,
            padding="pre"
        )

        # Predict
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = int(np.argmax(predicted_probs))

        # Index to word
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        next_words.append(predicted_word)
        result += " " + predicted_word

    return PredictResponse(
        input_text=request.seed_text,
        predicted_text=result,
        next_words=next_words
    )
