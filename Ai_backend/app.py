import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json

# Load environment variables
load_dotenv()

# Configure Gemini - use environment variable for API key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

genai.configure(api_key=api_key)

# Use the correct model
model = genai.GenerativeModel("models/gemini-2.5-flash")

app = Flask(__name__)
CORS(app)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Read image bytes
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
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
        else:
            parsed = {
                "classification": "unknown",
                "description": "Could not extract valid JSON from response."
            }

        return jsonify(parsed)

    except Exception as e:
        print("Gemini API Error:", str(e))
        return jsonify({"error": "AI processing failed"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
