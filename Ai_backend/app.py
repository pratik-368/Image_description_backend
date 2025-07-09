import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import re
import json
from werkzeug.exceptions import RequestEntityTooLarge
from functools import wraps
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load .env variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_RETRIES = 3

# Initialize Gemini API with error handling
def initialize_gemini():
    """Initialize Gemini API with proper error handling"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        logger.info("Gemini API initialized successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Gemini API: {str(e)}")
        raise

# Global model instance
try:
    model = initialize_gemini()
except Exception as e:
    logger.critical(f"Application startup failed: {str(e)}")
    model = None

# Utility functions
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(file):
    """Validate uploaded image file"""
    if not file or file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size (additional check beyond Flask config)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if size > app.config['MAX_CONTENT_LENGTH']:
        return False, "File too large. Maximum size: 16MB"
    
    return True, "Valid file"

def parse_gemini_response(response_text):
    """Parse Gemini response with robust error handling"""
    try:
        # Try to extract JSON from response
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not match:
            logger.warning("No JSON found in Gemini response")
            return {
                "classification": "other",
                "description": "Unable to parse AI response"
            }
        
        parsed = json.loads(match.group())
        
        # Validate required fields
        if "classification" not in parsed or "description" not in parsed:
            logger.warning("Missing required fields in parsed response")
            return {
                "classification": "other",
                "description": "Incomplete AI response"
            }
        
        # Validate classification values
        valid_classifications = ["garbage", "pothole", "open drainage", "other"]
        if parsed["classification"].lower() not in valid_classifications:
            logger.warning(f"Invalid classification: {parsed['classification']}")
            parsed["classification"] = "other"
        
        return parsed
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {
            "classification": "other",
            "description": "Failed to parse AI response"
        }
    except Exception as e:
        logger.error(f"Unexpected error parsing response: {str(e)}")
        return {
            "classification": "other",
            "description": "Error processing AI response"
        }

def error_handler(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }), 500
    return decorated_function

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "error": "Bad request",
        "message": "The request could not be understood by the server"
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Not found",
        "message": "The requested resource was not found"
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "error": "Method not allowed",
        "message": "The method is not allowed for the requested URL"
    }), 405

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({
        "error": "File too large",
        "message": "The uploaded file exceeds the maximum size limit (16MB)"
    }), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check if Gemini API is available
        if model is None:
            return jsonify({
                "status": "unhealthy",
                "message": "Gemini API not initialized"
            }), 503
        
        return jsonify({
            "status": "healthy",
            "message": "Service is running"
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "message": "Service check failed"
        }), 503

# Main classification endpoint
@app.route('/classify', methods=['POST'])
@error_handler
def classify_image():
    """Classify uploaded image with comprehensive error handling"""
    
    # Check if Gemini API is available
    if model is None:
        logger.error("Gemini API not available")
        return jsonify({
            "error": "Service unavailable",
            "message": "AI service is not available"
        }), 503
    
    # Validate request
    if 'image' not in request.files:
        logger.warning("No image in request")
        return jsonify({
            "error": "No image provided",
            "message": "Please upload an image file"
        }), 400
    
    image_file = request.files['image']
    
    # Validate file
    is_valid, message = validate_image_file(image_file)
    if not is_valid:
        logger.warning(f"Invalid file: {message}")
        return jsonify({
            "error": "Invalid file",
            "message": message
        }), 400
    
    try:
        # Read image data
        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({
                "error": "Empty file",
                "message": "The uploaded file is empty"
            }), 400
        
        # Prepare prompt
        prompt = (
            "This image shows a civic problem in an urban area. "
            "Classify the issue as either garbage, pothole, open drainage, or other. "
            "Then give a 1-2 line description in JSON format like:\n"
            "{\"classification\": \"\", \"description\": \"\"}"
        )
        
        # Call Gemini API with retries
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"Calling Gemini API, attempt {attempt + 1}")
                response = model.generate_content([
                    prompt, 
                    {"mime_type": "image/jpeg", "data": image_bytes}
                ])
                
                if not response or not response.text:
                    raise Exception("Empty response from Gemini API")
                
                logger.info("Gemini API call successful")
                logger.debug(f"Raw response: {response.text}")
                
                # Parse response
                parsed = parse_gemini_response(response.text)
                
                return jsonify({
                    "success": True,
                    "classification": parsed["classification"],
                    "description": parsed["description"]
                })
                
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini API attempt {attempt + 1} failed: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    continue
                else:
                    break
        
        # If all retries failed
        logger.error(f"All Gemini API attempts failed. Last error: {str(last_error)}")
        return jsonify({
            "error": "AI processing failed",
            "message": "Unable to process image after multiple attempts"
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in classify_image: {str(e)}")
        return jsonify({
            "error": "Processing error",
            "message": "An error occurred while processing the image"
        }), 500

# Root endpoint
@app.route('/', methods=['GET'])
def root():
    """Root endpoint with service info"""
    return jsonify({
        "service": "Civic Problem Classifier",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/classify (POST)",
            "health": "/health (GET)"
        }
    })

# Application startup
if __name__ == '__main__':
    # Get port from environment (Render sets this)
    port = int(os.environ.get("PORT", 10000))  # fallback to 10000 as per Render default
   
    
    # Check if we're in production
    debug_mode = os.environ.get("FLASK_ENV") != "production"
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug_mode}")
    
    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug_mode
        )
    except Exception as e:
        logger.critical(f"Failed to start Flask app: {str(e)}")
        raise
