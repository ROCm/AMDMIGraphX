import torch
from diffusers import AutoencoderKL
import os


class VAEDecoder(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent_sample):
        return self.vae.decode(latent_sample)


vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix")
vae.eval()
output = "models/sdxl-1.0-base/vae_decoder_fp16_fix/model.onnx"
os.makedirs(os.path.dirname(output), exist_ok=True)
torch.onnx.export(VAEDecoder(vae),
                  torch.randn(1, 4, 128, 128),
                  output,
                  export_params=True,
                  do_constant_folding=True,
                  input_names=['latent_sample'])
