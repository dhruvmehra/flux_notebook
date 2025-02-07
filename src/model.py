import torch
from diffusers import FluxPipeline, StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
import insightface
import cv2
from .config import Config

class FluxModel:
    def __init__(self):
        self.device = torch.device(Config.get_device())
        self.face_analyzer = self._setup_face_analyzer()
        self.flux_pipe = self._load_model(
            Config.FLUX_MODEL_ID, 
            torch.bfloat16,
            enable_cpu_offload=True
        )
        self.inpaint_pipe = self._load_model(
            Config.INPAINT_MODEL_ID, 
            torch.float16
        )
    
    def _setup_face_analyzer(self):
        analyzer = insightface.app.FaceAnalysis()
        analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        return analyzer
    
    def _load_model(self, model_id, dtype, enable_cpu_offload=False):
        """Generic model loader for both FLUX and Inpainting models"""
        pipe_class = FluxPipeline if "flux" in model_id.lower() else StableDiffusionInpaintPipeline
        pipe = pipe_class.from_pretrained(
            model_id,
            torch_dtype=dtype if self.device.type == "cuda" else torch.float32,
            cache_dir=Config.MODEL_CACHE
        )
        
        if self.device.type == "cuda":
            pipe = pipe.to(self.device)
            if enable_cpu_offload:
                pipe.enable_model_cpu_offload()
        
        return pipe
    
    def process_face(self, image):
        """Process and extract face features"""
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        faces = self.face_analyzer.get(image)
        return faces[0] if faces else None
    
    def generate_with_face(self, face_image, prompt, num_steps=Config.DEFAULT_STEPS):
        """Generate image using face reference"""
        face_features = self.process_face(face_image)
        if face_features is None:
            return None
        
        return self.flux_pipe(
            prompt=f"A photo of a person with similar features to the reference, {prompt}",
            num_inference_steps=num_steps
        ).images[0]
    
    def inpaint(self, image, mask, prompt, num_steps=Config.DEFAULT_STEPS):
        """Perform inpainting on image"""
        if not all([image, mask]):
            return None
        
        # Convert to PIL if needed
        image = Image.fromarray(image) if not isinstance(image, Image.Image) else image
        mask = Image.fromarray(mask) if not isinstance(mask, Image.Image) else mask
        
        # Resize for inpainting
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        
        return self.inpaint_pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_steps
        ).images[0] 