import os
from pathlib import Path
import torch

class Config:
    # Project paths
    ROOT_DIR = Path(os.path.dirname(os.path.dirname(__file__)))
    MODEL_CACHE = ROOT_DIR / "model_cache"
    OUTPUT_DIR = ROOT_DIR / "outputs"
    
    # Model settings
    FLUX_MODEL_ID = "black-forest-labs/FLUX.1-dev"
    INPAINT_MODEL_ID = "runwayml/stable-diffusion-inpainting"  # Added for inpainting
    DEVICE = "cuda"  # or "cpu"
    
    # Default generation settings
    DEFAULT_STEPS = 50
    DEFAULT_GUIDANCE_SCALE = 3.5  # Default for Flux
    DEFAULT_SIZE = 1024  # Flux default size
    MAX_SEQUENCE_LENGTH = 512  # Flux specific parameter
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def is_testing(cls):
        """Check if we're in testing mode"""
        return os.environ.get('FLUX_TESTING', '0') == '1'

    @classmethod
    def get_device(cls):
        """Get appropriate device based on environment"""
        if cls.is_testing():
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu" 