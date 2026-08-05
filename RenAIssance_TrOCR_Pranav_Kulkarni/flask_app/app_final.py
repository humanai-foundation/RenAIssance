"""
Main application launcher for the modular OCR Flask application
This file replaces app2.py and provides the same functionality in a modular structure
"""
import os
import sys
import logging
from flask import Flask
from flask_cors import CORS

# Add the modular_app directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modular_app.config import get_config, setup_directories, print_config_summary
from modular_app.llm_service import LLMService
from modular_app.ocr_models import OCRModelsService
from modular_app.ocr_processing import OCRProcessingService
from modular_app.routes import create_routes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app() -> Flask:
    """Create and configure Flask application"""
    try:
        # Load configuration
        config = get_config()
        
        logger.info("Starting Modular OCR Flask Application...")
        
        # Print configuration summary
        print_config_summary(config)
        
        # Setup directories
        setup_directories(config)
        
        # Create Flask app
        app = Flask(__name__)
        app.config['UPLOAD_FOLDER'] = config.upload_folder
        app.config['MAX_CONTENT_LENGTH'] = config.max_file_size
        
        # Enable CORS
        CORS(app)
        
        # Initialize services
        logger.info("Initializing services...")
        
        # Initialize LLM service
        llm_service = LLMService(config)
        
        # Initialize OCR models service
        ocr_models_service = OCRModelsService(config)
        
        # Initialize OCR processing service
        ocr_processing_service = OCRProcessingService(config, ocr_models_service, llm_service)
        
        # Pre-load the models on startup
        logger.info("Pre-loading models...")
        try:
            ocr_models_service.load_textline_model()
            ocr_models_service.load_trocr_model()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not pre-load models: {e}")
        
        # Test the LLM fallback system
        try:
            llm_service.manual_fallback_test()
        except Exception as e:
            logger.warning(f"LLM fallback test failed: {e}")
        
        # Register routes
        logger.info("Registering routes...")
        routes_bp = create_routes(config, ocr_processing_service, llm_service)
        app.register_blueprint(routes_bp)
        
        logger.info("Flask application created successfully")
        return app
        
    except Exception as e:
        logger.error(f"Failed to create Flask application: {e}")
        raise

def main():
    """Main function to run the application"""
    app = create_app()
    
    # Print startup information
    print("=" * 60)
    print("MODULAR OCR APPLICATION STARTED")
    print("=" * 60)
    print("Application is running at: http://localhost:5000")
    print("Upload folder:", app.config['UPLOAD_FOLDER'])
    print("Max file size:", app.config['MAX_CONTENT_LENGTH'] / (1024*1024), "MB")
    print("=" * 60)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
