import torch
from diffusers import FluxPipeline
from huggingface_hub import login
from diffusers import DiffusionPipeline

class PipeProvider:
    def __init__(self) -> None:
        pass

    def load_pipe(self):
        print("Logging into hugging face")
        login(token="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")
        print("Calling load_pipe...")
        pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",torch_dtype=torch.bfloat16, device_map="balanced")
        pipe.save_pretrained("wsgi_runtime/model")
        # pipe = FluxPipeline.from_pretrained("models/lumina_model", torch_dtype=torch.bfloat16, device_map="balanced")        
        return pipe