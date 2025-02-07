import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
from .config import Config

class FluxModel:
    def __init__(self):
        self.device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
        self.pipe = self._load_model()
    
    def _load_model(self):
        """Load the Flux model"""
        pipe = FluxPipeline.from_pretrained(
            Config.FLUX_MODEL_ID,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            cache_dir=Config.MODEL_CACHE
        )
        
        if self.device.type == "cuda":
            pipe.enable_model_cpu_offload()  # Optimize VRAM usage
        
        return pipe
    
    def text_to_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = Config.DEFAULT_SIZE,
        height: int = Config.DEFAULT_SIZE,
        num_inference_steps: int = Config.DEFAULT_STEPS,
        guidance_scale: float = Config.DEFAULT_GUIDANCE_SCALE,
        seed: int = None
    ) -> Image.Image:
        """Generate image from text prompt using Flux"""
        generator = None
        if seed is not None:
            generator = torch.Generator("cpu").manual_seed(seed)
        
        output = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=Config.MAX_SEQUENCE_LENGTH,
            generator=generator
        ).images[0]
        
        return output 