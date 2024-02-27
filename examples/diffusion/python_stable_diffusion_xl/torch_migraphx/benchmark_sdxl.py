import copy
import torch
from diffusers import DiffusionPipeline
import torch_migraphx


def benchmark(func, iters, *args, **kwargs):
    # Warm up
    for _ in range(1):
        func(*args, **kwargs)

    # Start benchmark.
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        out = func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # in ms
    return (start_event.elapsed_time(end_event)) / iters


if __name__ == '__main__':
    # torch.random.manual_seed(10)
    model_repo = 'stabilityai/stable-diffusion-xl-base-1.0'
    prompts = [
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    ]
    num_steps = 30
    fname = 'benchmark_output.png'

    pipe = DiffusionPipeline.from_pretrained(model_repo,
                                             torch_dtype=torch.float16,
                                             use_safetensors=True,
                                             variant="fp16").to("cuda")
    # pipe = DiffusionPipeline.from_pretrained(model_repo).to("cuda")

    pipe.text_encoder = torch.compile(
        pipe.text_encoder,
        backend='migraphx',
        options={
            # "save_mxr": True,
            # "save_compiled": "text_encoder.pt",
            # "exhaustive_tune": True,
        },
    )

    # pipe.text_ecoder_2 = torch.compile(pipe.text_encoder_2, backend='migraphx')

    pipe.unet = torch.compile(
        pipe.unet,
        backend='migraphx',
        options={
            # "save_mxr": True,
            # "save_compiled": "unet_attn.pt",
            # "exhaustive_tune": True,
        },
    )

    # pipe.vae.decoder = torch.compile(
    #     pipe.vae.decoder,
    #     backend='migraphx',
    #     options={"load_compiled": "decoder.pt"},
    # )

    inputs = {
        "prompt": prompts,
        "height": 1024,
        "width": 1024,
        "num_inference_steps": num_steps,
        "num_images_per_prompt": 1,
    }
    image = pipe(**inputs).images[0]
    image.save(fname)

    print("Benchmarking...")
    t = benchmark(pipe, 10, **inputs)

    print(f"sd e2e: {t} ms")
