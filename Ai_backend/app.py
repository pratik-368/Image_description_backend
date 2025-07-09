import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json

# Load .env variables
load_dotenv()

# Configure Gemini API using environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Use the Gemini model
model = genai.GenerativeModel("models/gemini-2.5-flash")

app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    prompt = (
        "This image shows a civic problem in an urban area. "
        "Classify the issue as either garbage, pothole, open drainage, or other. "
        "Then give a 1-2 line description in JSON format like:\n"
        "{\"classification\": \"\", \"description\": \"\"}"
    )

    try:
        response = model.generate_content(
            [prompt, {"mime_type": "image/jpeg", "data": image_bytes}]
        )

        print("Gemini raw response:\n", response.text)

        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {
            "classification": "Unknown",
            "description": "Could not extract valid JSON."
        }

        return jsonify(parsed)

    except Exception as e:
        print("Gemini API Error:", str(e))
        return jsonify({"error": "AI processing failed"}), 500

# 🚀 Critical for Render — bind to 0.0.0.0 and use PORT from environment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets this PORT
    app.run(host='0.0.0.0', port=port)
