# For sample predict functions for popular libraries visit https://github.com/opendatahub-io/odh-prediction-samples

# Import libraries
# import tensorflow as tf

import torch
from diffusers import FluxPipeline
from PIL import Image
import base64
import io

# Load your model.
# model_dir = 'models/myfancymodel'
# saved_model = tf.saved_model.load(model_dir)
# predictor = saved_model.signatures['default']


# Write a predict function 
def predict(args_dict,pipe : FluxPipeline):
#     arg = args_dict.get('arg1')
#     predictor(arg)
    prompt = args_dict.get('prompt')
    print("Lumina : Prompt:"+prompt)
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator().manual_seed(0)
    ).images[0]
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return {'prediction': encoded_image}

