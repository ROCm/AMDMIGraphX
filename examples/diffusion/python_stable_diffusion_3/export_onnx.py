import torch
from diffusers import StableDiffusion3Pipeline
import os

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# pipe = pipe.to("cuda")
# print(pipe)
# print(pipe.text_encoder)
x=torch.randint(1, (1, 77))
# pipe.text_encoder.eval()
output_path='models'
encoder_path=output_path+'/text_encoder/text_encoder.onnx'
encoder_2_path=output_path+'/text_encoder_2/text_encoder_2.onnx'
encoder_3_path=output_path+'/text_encoder_3/text_encoder_3.onnx'
print(output_path)
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
os.makedirs(os.path.dirname(encoder_path), exist_ok=True)
os.makedirs(os.path.dirname(encoder_2_path), exist_ok=True)
os.makedirs(os.path.dirname(encoder_3_path), exist_ok=True)

torch.onnx.export(pipe.text_encoder,
                      x,
                      encoder_path,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input_ids'],
                      dynamic_axes={'input_ids': { 0: 'batch_size'}})
torch.onnx.export(pipe.text_encoder_2,
                      x,
                      encoder_2_path,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input_ids'],
                      dynamic_axes={'input_ids': { 0: 'batch_size'}})
torch.onnx.export(pipe.text_encoder_3,
                      x,
                      encoder_3_path,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['input_ids'],
                      dynamic_axes={'input_ids': { 0: 'batch_size'}})


# export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
# onnx_program = torch.onnx.dynamo_export(
#     pipe.text_encoder,
#     *x,
#     # **kwargs,
#     export_options=export_options)
# onnx_program.save("text_encoder.onnx")
