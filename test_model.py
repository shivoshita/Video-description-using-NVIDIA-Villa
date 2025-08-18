import os
import torch
from transformers import VideoMAEForVideoClassification, AutoProcessor

def check_model():
    print("Checking VideoMAE model...")
    
    # Check if local model exists
    local_path = "models/videomae/"
    if os.path.exists(local_path):
        print(f"Local model found at: {local_path}")
        try:
            # Check model files
            files = os.listdir(local_path)
            print(f"Model files: {files}")
            
            # Try loading
            model = VideoMAEForVideoClassification.from_pretrained(local_path)
            processor = AutoProcessor.from_pretrained(local_path)
            
            print("✅ Local model loaded successfully")
            print(f"Model config: {model.config}")
            
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            return False
    else:
        print("❌ Local model not found")
        return False
    
    return True

def test_fallback_model():
    print("\nTesting fallback model...")
    try:
        model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        processor = AutoProcessor.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        print("✅ Fallback model loaded successfully")
        print(f"Labels available: {len(model.config.id2label)}")
        return True
    except Exception as e:
        print(f"❌ Error loading fallback model: {e}")
        return False

if __name__ == "__main__":
    local_ok = check_model()
    if not local_ok:
        print("Trying fallback model...")
        test_fallback_model()