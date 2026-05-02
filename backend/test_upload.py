"""
Test script for NeuroLens upload functionality
"""
import requests
import json
from PIL import Image
import io
import os

# Configuration
BACKEND_URL = "http://localhost:5000"
TEST_IMAGE_SIZE = (224, 224)

def create_test_image():
    """Create a simple test image"""
    print("📷 Creating test image...")
    img = Image.new('RGB', TEST_IMAGE_SIZE, color=(73, 109, 137))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes, 'test_image.png'

def test_health():
    """Test health endpoint"""
    print("\n🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BACKEND_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check passed")
            print(f"  - Status: {data['status']}")
            print(f"  - Models loaded: {data['models_loaded']}")
            print(f"  - Device: {data['device']}")
            return True
        else:
            print(f"✗ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to backend at http://localhost:5000")
        print("  Make sure Flask server is running: python app.py")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False

def test_tumor_detection():
    """Test tumor detection endpoint"""
    print("\n🧠 Testing tumor detection upload...")
    try:
        img_bytes, filename = create_test_image()
        
        files = {'file': (filename, img_bytes, 'image/png')}
        response = requests.post(f"{BACKEND_URL}/detect-tumor", files=files)
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✓ Upload successful!")
                print(f"  - Diagnosis: {data['diagnosis']}")
                print(f"  - Confidence: {data['confidence']:.1f}%")
                print(f"  - Status: {data['status']}")
                print(f"  - Probabilities:")
                for cls, prob in data['probabilities'].items():
                    print(f"    • {cls}: {prob:.1f}%")
                return True
            else:
                print(f"✗ Upload failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"✗ Upload failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Upload error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_validation():
    """Test file validation"""
    print("\n📋 Testing file validation...")
    
    # Test invalid file extension
    print("  Testing invalid extension...")
    invalid_file = io.BytesIO(b"not an image")
    files = {'file': ('test.txt', invalid_file, 'text/plain')}
    response = requests.post(f"{BACKEND_URL}/detect-tumor", files=files)
    if response.status_code == 400:
        print("  ✓ Invalid file type rejected")
    else:
        print(f"  ✗ Expected 400, got {response.status_code}")
    
    # Test empty file
    print("  Testing empty form...")
    response = requests.post(f"{BACKEND_URL}/detect-tumor")
    if response.status_code == 400:
        print("  ✓ Empty request rejected")
    else:
        print(f"  ✗ Expected 400, got {response.status_code}")

if __name__ == '__main__':
    print("="*50)
    print("NeuroLens Upload Functionality Test")
    print("="*50)
    
    # Run tests
    health_ok = test_health()
    
    if health_ok:
        test_tumor_detection()
        test_file_validation()
    else:
        print("\n⚠️  Backend is not responding. Start it with: python app.py")
    
    print("\n" + "="*50)
    print("Test complete!")
    print("="*50)
