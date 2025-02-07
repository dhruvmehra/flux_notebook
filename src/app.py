import gradio as gr
from .model import FluxModel
from .config import Config

class FluxApp:
    def __init__(self):
        self.model = FluxModel()
        Config.setup_directories()
    
    def create_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("# 🎨 AI Image Generation & Editing")
            
            with gr.Tabs():
                self._create_generation_tab()
                self._create_inpainting_tab()
        
        return demo
    
    def _create_generation_tab(self):
        with gr.Tab("Generate with Face"):
            with gr.Row():
                with gr.Column():
                    face_input = gr.Image(label="Face Reference")
                    gen_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the scene (e.g., 'in a cyberpunk city')"
                    )
                    gen_steps = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Steps"
                    )
                    gen_button = gr.Button("🎨 Generate")
                gen_output = gr.Image(label="Generated Image")
            
            gen_button.click(
                fn=self.model.generate_with_face,
                inputs=[face_input, gen_prompt, gen_steps],
                outputs=gen_output
            )
    
    def _create_inpainting_tab(self):
        with gr.Tab("Inpainting"):
            with gr.Row():
                with gr.Column():
                    inpaint_input = gr.Image(label="Image to Edit")
                    inpaint_mask = gr.Image(
                        label="Draw Mask",
                        tool="sketch"
                    )
                    inpaint_prompt = gr.Textbox(
                        label="What to Add",
                        placeholder="Describe what to add in masked area"
                    )
                    inpaint_steps = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Steps"
                    )
                    inpaint_button = gr.Button("🖌️ Edit")
                inpaint_output = gr.Image(label="Result")
            
            inpaint_button.click(
                fn=self.model.inpaint,
                inputs=[inpaint_input, inpaint_mask, inpaint_prompt, inpaint_steps],
                outputs=inpaint_output
            )
    
    def launch(self, **kwargs):
        demo = self.create_interface()
        demo.launch(**kwargs) 