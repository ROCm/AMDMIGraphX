from argparse import ArgumentParser
from flux_pipeline import FluxPipeline


def get_args():
    parser = ArgumentParser()
    
    parser.add_argument(
        "--hf-model",
        type=str,
        choices=["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell",],
        default="black-forest-labs/FLUX.1-dev",
        help="Specify HF model card. Options: 'black-forest-labs/FLUX.1-dev', 'black-forest-labs/FLUX.1-schnell'",
    )
    
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Specify directory with local onnx files (or where to export)",
    )
    
    parser.add_argument(
        "--compile-dir",
        type=str,
        default=None,
        help="Specify directory with compile mxr files (or where to export)",
    )
    
    parser.add_argument(
        "-d",
        "--image-height",
        type=int,
        default=1024,
        help="Output Image height, default 1024",
    )
    
    parser.add_argument(
        "-w",
        "--image-width",
        type=int,
        default=1024,
        help="Output Image width, default 1024",
    )
    
    parser.add_argument(
        "-g",
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale, default 3.5",
    )
    
    parser.add_argument(
        "-l",
        "--max-sequence-length",
        type=int,
        default=512,
        help="Max sequence length for T5, default 512",
    )
    
    parser.add_argument(
        "-p",
        "--prompt",
        default=["A cat holding a sign that says hello world"],
        nargs="*",
        help="Text prompt(s) to be sent to the CLIP tokenizer and text encoder",
    )
    
    parser.add_argument(
        "--prompt2",
        default=None,
        nargs="*",
        help="Text prompt(s) to be sent to the T5 tokenizer and text encoder. If not defined, prompt will be used instead",
    )
    
    parser.add_argument(
        "-s",
        "--denoising-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    
    parser.add_argument(
        "--fp16", 
        action='store_true', 
        help="Apply fp16 quantization."
    )
    
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Specify directory where images should be saved",
    )
    
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="flux",
        help="Specify image name prefix for saving result images",
    )
    
    parser.add_argument(
        "-b",
        "--benchmark-runs",
        type=int,
        default=None,
        help="Number of runs to do for benchmarking. Default: no benchmarking",
    )
    
    parser.add_argument(
        "--exhaustive-tune",
        action='store_true', 
        help="Perform exhaustive tuning when compiling"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Set custom batch size (expects len 1 prompt, useful for benchmarking)"
    )
    
    return parser.parse_args()
    

if __name__ == "__main__":
    args = get_args()
    
    prompt = args.prompt
    prompt2 = args.prompt2 if args.prompt2 else prompt
    
    if args.batch_size:
        assert len(prompt) == 1 and len(prompt2) == 1
        prompt = prompt * args.batch_size
        prompt2 = prompt2 * args.batch_size
      
    pipe = FluxPipeline(
        hf_model_path=args.hf_model,
        local_dir=args.local_dir,
        compile_dir=args.compile_dir,
        img_height=args.image_height,
        img_width=args.image_width,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        batch_size=len(prompt),
        denoising_steps=args.denoising_steps,
        fp16=args.fp16,
        exhaustive_tune=args.exhaustive_tune
    )
    
    pipe.load_models()
    
    images = pipe.infer(prompt, prompt2, warmup=True)
    
    if args.output_dir:
        print(f"Saving images to {args.output_dir}")
        pipe.save_image(images, args.output_prefix, args.output_dir)
    
    if args.benchmark_runs:
        pipe.clear_run_data()
        print("Begin benchmarking...")
        for _ in range(args.benchmark_runs):
            pipe.infer(prompt, prompt2)
            print(f"Run time: {pipe.times[-1]}s")
            
        pipe.print_summary()