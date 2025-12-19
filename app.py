from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

@app.route("/")
def home():
    return "Sentiment Analysis API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "text" not in data:
        return jsonify({"error": "Please provide text"}), 400

    text = data["text"]
    result = sentiment_analyzer(text)

    return jsonify({
        "text": text,
        "sentiment": result
    })

if __name__ == "__main__":
    app.run(debug=True)
