from argparse import ArgumentParser

import torch
from migx_sdxl_pipeline import MGXSDXLPipeline

parser = ArgumentParser()
parser.add_argument(
    "--hf-model-path",
    type=str,
    default="stabilityai/stable-diffusion-xl-base-1.0",
    help="Huggingface repo path",
)

parser.add_argument(
    "--torch-seed",
    type=int,
    default=None,
    help="Set global torch seed",
)

parser.add_argument(
    "--denoise-steps",
    type=int,
    default=30,
    help="Set global torch seed",
)

parser.add_argument(
    "--exhaustive-tune",
    action="store_true",
    default=False,
    help="Perform exhaustive tuning when compiling onnx models",
)

parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    default=
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    help="Prompt",
)

parser.add_argument(
    "-n",
    "--negative-prompt",
    type=str,
    default="",
    help="Negative prompt",
)

parser.add_argument(
    "--guidance-scale",
    type=float,
    default=5.0,
    help="Guidance scale",
)

parser.add_argument(
    "--onnx-model-dir",
    type=str,
    default="/data/stable-diffusion-xl-1.0-tensorrt/sdxl-1.0-base",
    help="Path to onnx model exports",
)

parser.add_argument(
    "--compiled-model-dir",
    type=str,
    default="./sdxl-1.0-base",
    help="Path to onnx model exports",
)

parser.add_argument(
    "--load-compiled",
    action="store_true",
    default=False,
    help="Load compiled models from compiled-model-dir",
)

parser.add_argument(
    "--save-compiled",
    action="store_true",
    default=False,
    help="Save compiled models to compiled-model-dir",
)

parser.add_argument(
    "--quantize-fp16",
    action="store_true",
    default=False,
    help="Save compiled models to compiled-model-dir",
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.torch_seed:
        torch.manual_seed(args.torch_seed)

    mgx_pipe = MGXSDXLPipeline(base_model_path=args.hf_model_path,
                               denoising_steps=args.denoise_steps,
                               guidance_scale=args.guidance_scale,
                               quantize_fp16=args.quantize_fp16,
                               exhaustive_tune=args.exhaustive_tune,
                               seed=args.torch_seed)

    if args.load_compiled:
        # TODO: Support loading compiled mxr
        mgx_pipe.load_models(args.compiled_model_dir, file_type="pt")
    else:
        mgx_pipe.load_models(args.onnx_model_dir, file_type="onnx")

    if args.save_compiled:
        mgx_pipe.save_compiled_models(args.compiled_model_dir)

    images = mgx_pipe.run(args.prompt, args.negative_prompt)

    mgx_pipe.save_image(images, "./")
    mgx_pipe.print_summary()
