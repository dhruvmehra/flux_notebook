import os
os.environ['FLUX_TESTING'] = '1'  # Enable test mode

from src.model import FluxModel
from PIL import Image

def test_model_loading():
    """Test if model can be loaded"""
    try:
        model = FluxModel()
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        return None

def test_simple_generation(model):
    """Test basic image generation"""
    if model is None:
        return
    
    try:
        prompt = "A simple test image of a blue circle"
        image = model.text_to_image(
            prompt=prompt,
            width=512,  # Smaller size for testing
            height=512,
            num_inference_steps=20  # Fewer steps for testing
        )
        
        if isinstance(image, Image.Image):
            image.save("test_output.png")
            print("✅ Image generation successful")
            print("📝 Saved as test_output.png")
        else:
            print("❌ Image generation failed")
    except Exception as e:
        print(f"❌ Generation error: {str(e)}")

if __name__ == "__main__":
    print("🔍 Starting local FLUX model test...")
    print("Note: This will run in CPU mode and might be slow")
    
    model = test_model_loading()
    test_simple_generation(model) 