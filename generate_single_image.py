"""Generate a single image for a given class using a specified pipeline model."""
import argparse
import gc
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_idx', type=int, required=True)
    parser.add_argument('--pipeline', type=str, required=True, choices=['2', '3'])
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    args = parser.parse_args()

    AMUSED_PROMPTS = {
        0: "a golden retriever dog, white background, photorealistic",
        1: "a colorful bird perched on a branch, white background, photorealistic",
        2: "a tropical fish swimming, white background, photorealistic",
        3: "a pentagram star shape, black lines on white background",
        4: "a square shape, black lines on white background",
        5: "a circle shape, black lines on white background",
        6: "a pair of scissors, product photo, white background",
        7: "a wristwatch, product photo, white background",
        8: "a ceramic coffee cup, product photo, white background",
        9: "a wooden chair, product photo, white background",
    }

    SD_PROMPTS = {
        0: "a golden retriever dog, full body, studio white background, photorealistic, 8k, sharp focus, professional photography",
        1: "a colorful tropical bird perched on a branch, studio white background, photorealistic, 8k, sharp focus, professional photography",
        2: "a tropical fish swimming, studio white background, photorealistic, 8k, sharp focus, professional photography",
        3: "a pentagram star shape drawn with thick black lines, clean white background, vector art, sharp, high contrast",
        4: "a perfect square drawn with thick black lines, clean white background, vector art, sharp, high contrast",
        5: "a perfect circle drawn with thick black lines, clean white background, vector art, sharp, high contrast",
        6: "a pair of scissors, product photography, studio white background, photorealistic, 8k, sharp focus",
        7: "a wristwatch with metal band, product photography, studio white background, photorealistic, 8k, sharp focus",
        8: "a white ceramic coffee cup, product photography, studio white background, photorealistic, 8k, sharp focus",
        9: "a wooden dining chair, product photography, studio white background, photorealistic, 8k, sharp focus",
    }

    SD_NEGATIVE = "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, logo, oversaturated, cartoon, anime, sketch, noisy"

    device = f'cuda:{args.cuda}'
    gc.collect()
    torch.cuda.empty_cache()

    if args.pipeline == '2':
        from diffusers import AmusedPipeline
        pipe = AmusedPipeline.from_pretrained(
            "amused/amused-512", variant="fp16",
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
        )
        pipe.enable_model_cpu_offload()
        generator = torch.Generator(device=device).manual_seed(args.seed + args.class_idx)
        img = pipe(
            AMUSED_PROMPTS[args.class_idx],
            num_inference_steps=12, guidance_scale=10.0,
            generator=generator,
        ).images[0]
    else:
        from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16, safety_checker=None,
            low_cpu_mem_usage=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()
        generator = torch.Generator(device=device).manual_seed(args.seed + args.class_idx)
        img = pipe(
            SD_PROMPTS[args.class_idx],
            negative_prompt=SD_NEGATIVE,
            num_inference_steps=25, guidance_scale=7.5,
            height=512, width=512, generator=generator,
        ).images[0]

    img.save(args.output_path)
    print(f"Saved: {args.output_path}")


if __name__ == '__main__':
    main()
