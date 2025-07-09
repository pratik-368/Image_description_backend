import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Configure Gemini
genai.configure(api_key="GEMINI_API_KEY")

# Use the correct model from your list
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

        # Extract JSON from response
        import re, json
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        parsed = json.loads(match.group()) if match else {
            "classification": "Unknown",
            "description": "Could not extract valid JSON."
        }

        return jsonify(parsed)

    except Exception as e:
        print("Gemini API Error:", str(e))
        return jsonify({"error": "AI processing failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)
