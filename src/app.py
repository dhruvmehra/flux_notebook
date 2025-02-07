import gradio as gr
from .model import FluxModel
from .config import Config

class FluxApp:
    def __init__(self):
        self.model = FluxModel()
        Config.setup_directories()
    
    def create_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("# 🎨 FLUX.1 Image Generation Studio")
            
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="What you don't want to see...",
                    lines=2
                )
                
                with gr.Row():
                    with gr.Column():
                        width = gr.Slider(512, 2048, value=Config.DEFAULT_SIZE, step=64, label="Width")
                        height = gr.Slider(512, 2048, value=Config.DEFAULT_SIZE, step=64, label="Height")
                    
                    with gr.Column():
                        steps = gr.Slider(1, 100, value=Config.DEFAULT_STEPS, step=1, label="Steps")
                        guidance = gr.Slider(1, 20, value=Config.DEFAULT_GUIDANCE_SCALE, step=0.1, label="Guidance Scale")
                
                seed = gr.Number(label="Seed (optional)", precision=0)
                generate_button = gr.Button("🎨 Generate")
            
            output_image = gr.Image(label="Generated Image")
            
            generate_button.click(
                fn=self.model.text_to_image,
                inputs=[
                    prompt,
                    negative_prompt,
                    width,
                    height,
                    steps,
                    guidance,
                    seed
                ],
                outputs=output_image
            )
            
            gr.Markdown("""
                ### Tips:
                - FLUX.1 works best with detailed, clear prompts
                - The default guidance scale of 3.5 is optimized for FLUX
                - Higher step counts (50+) generally give better results
                - Use seed for reproducible results
                
                ### Note:
                This implementation uses the FLUX.1 [dev] model from Black Forest Labs, 
                a 12 billion parameter rectified flow transformer.
            """)
        
        return demo
    
    def launch(self, **kwargs):
        demo = self.create_interface()
        demo.launch(**kwargs) 