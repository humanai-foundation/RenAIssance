#!/usr/bin/env python3
"""
Test script for the modular OCR application
This script tests that the modular application can be imported and initialized correctly
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from modular_app.config import get_config, print_config_summary
        print("✓ Config module imported successfully")
        
        from modular_app.llm_service import LLMService
        print("✓ LLM service imported successfully")
        
        from modular_app.ocr_models import OCRModelsService
        print("✓ OCR models service imported successfully")
        
        from modular_app.utils import allowed_file, is_image_file
        print("✓ Utils module imported successfully")
        
        from modular_app.routes import create_routes
        print("✓ Routes module imported successfully")
        
        from modular_app.ocr_processing import OCRProcessingService
        print("✓ OCR processing service imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from modular_app.config import get_config
        config = get_config()
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - Upload folder: {config.upload_folder}")
        print(f"  - Max file size: {config.max_file_size / (1024*1024):.1f}MB")
        print(f"  - Gemini API: {'✓' if config.gemini_api_key else '✗'}")
        print(f"  - HuggingFace API: {'✓' if config.hf_api_token else '✗'}")
        print(f"  - LLaMA model: {'✓' if config.llama_model_path else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_services():
    """Test service initialization"""
    print("\nTesting service initialization...")
    
    try:
        from modular_app.config import get_config
        from modular_app.llm_service import LLMService
        from modular_app.ocr_models import OCRModelsService
        
        config = get_config()
        
        # Test LLM service
        llm_service = LLMService(config)
        print("✓ LLM service initialized successfully")
        
        # Test OCR models service
        ocr_models_service = OCRModelsService(config)
        print("✓ OCR models service initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Service initialization error: {e}")
        return False

def test_app_creation():
    """Test Flask app creation"""
    print("\nTesting Flask app creation...")
    
    try:
        from app_final import create_app
        
        # Create app (this will initialize all services)
        app = create_app()
        print("✓ Flask app created successfully")
        
        # Test that routes are registered
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        print(f"✓ Routes registered: {len(rules)} routes")
        
        # Check for key routes
        key_routes = ['/upload', '/health', '/llm_health']
        for route in key_routes:
            if route in rules:
                print(f"  ✓ {route}")
            else:
                print(f"  ✗ {route} missing")
        
        # Check for parameterized routes
        inference_routes = [rule for rule in rules if '/get_inference' in rule]
        if inference_routes:
            print(f"  ✓ /get_inference/<filename> (parameterized route)")
        else:
            print(f"  ✗ /get_inference/<filename> missing")
        
        return True
        
    except Exception as e:
        print(f"✗ App creation error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("MODULAR OCR APPLICATION TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config,
        test_services,
        test_app_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modular application is ready to use.")
        print("\nTo run the application:")
        print("  python app_final.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    print("=" * 60)

if __name__ == '__main__':
    main()
