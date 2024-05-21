import argparse
import os
import torch
import logging

from datasets.custom import CustomDataset, collate_fn
from datasets.utils import prepare_prompt
from models.infer import run_inference
from models.modeling_utils import load_models
from utils.image_utils import to_pil, denormalize, save_images_grid

PROMPTS = ['{} as a pornstar with big tits',
            'A photo of {}',
           '{} in Ghilbi anime style',
           '{} in Disney/Pixar style',
           '{} wears a red hat',
           '{} on the beach',
           'Manga drawing of {}',
           '{} as a Funko Pop figure',
            'Latte art of {}',
            '{} flower arrangement',
           'Pointillism painting of {}',
           '{} stained glass window',
           '{} is camping in the mountains',
           '{} is a character in a video game',
           'Watercolor painting of {}',
           '{} as a knight in plate',
           '{} as a character in a comic book',]

PROMPTS_NAMES = ['pornstar','photo','ghibli', 'disney_pixar', 'red_hat', 'beach', 'manga', 'funko_pop', 'latte_art', 'flower_arrangement', 'pointillism', 'stained_glass', 'camping', 'video_game', 'watercolor', 'knight', 'comic_book']

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_photoverse_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default=None,
        required=True,
        help="Training datasets root path",
    )
    parser.add_argument(
        "--img_subfolder",
        type=str,
        default="images",
        help="Subfolder relative to data_root_path containing images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for datasets loading. 0 means that the datasets will be loaded in the main process."
        ),
    )
    parser.add_argument(
        '--denoise_timesteps',
        type=int,
        default=10,
        help='Number of timesteps for inference'
    )

    parser.add_argument(
        '--guidance_scale',
        type=float,
        default=2.0,
        help='Guidance scale for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)"
    )
    parser.add_argument("--extra_num_tokens", type=int, default=4, help="Number of additional tokens")
    parser.add_argument(
        "--image_encoder_layers_idx",
        type=list,
        default=[4, 8, 12, 16],
        help="Image encoder extra layers indices to use as tokens for the text encoder, should be equal to extra_num_tokens",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--max_gen_images",
        type=int,
        default=300,
        help="Maximum number of generated images to save",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.pretrained_photoverse_path.split('/')[-1]
    model_name = model_name.split('.')[0]
    image_encoder_layers_idx = torch.tensor(args.image_encoder_layers_idx).to(args.device)

    tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter, scheduler, _ = load_models(
        args.pretrained_model_name_or_path, args.extra_num_tokens, args.pretrained_photoverse_path)

    vae.to(args.device)
    unet.to(args.device)
    text_encoder.to(args.device)
    image_encoder.to(args.device)
    image_adapter.to(args.device)
    text_adapter.to(args.device)

    for split in ["train", "test"]:
        dataset = CustomDataset(data_root=os.path.join(args.data_root_path,split), img_subfolder=args.img_subfolder,
                      tokenizer=tokenizer, size=args.resolution)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        full_output_dir = os.path.join(args.output_dir, f"{model_name}_{args.denoise_timesteps}_timesteps", split)
        os.makedirs(full_output_dir, exist_ok=True)
        for batch_idx, sample in enumerate(dataloader):
            if (batch_idx + 1)*args.batch_size > args.max_gen_images:
                break
            with torch.no_grad():
                pixel_values = sample["pixel_values"].to(args.device)
                input_images = [to_pil(denormalize(img)) for img in pixel_values]
                grid_data = [("Input Images", input_images)]
                for row_idx, input_image in enumerate(input_images):
                    os.makedirs(os.path.join(full_output_dir, f"grid_{batch_idx}_row_{row_idx}"), exist_ok=True)
                    input_image.save(os.path.join(full_output_dir, f"grid_{batch_idx}_row_{row_idx}", "input_image.png"))
                for prompt, prompt_name in zip(PROMPTS, PROMPTS_NAMES):
                    sample_to_update = prepare_prompt(tokenizer, prompt, "*",
                                                       num_of_samples=len(pixel_values))
                    sample.update(sample_to_update)
                    gen_tensors = run_inference(sample, tokenizer, image_encoder, text_encoder, unet, text_adapter,
                                                image_adapter, vae,
                                                scheduler, args.device, image_encoder_layers_idx,
                                                guidance_scale=args.guidance_scale,
                                                timesteps=args.denoise_timesteps, token_index=0)
                    gen_images = [to_pil(denormalize(gen_tensor)) for gen_tensor in gen_tensors]
                    for sample_idx, gen_image in enumerate(gen_images):
                        gen_image.save(os.path.join(full_output_dir, f"grid_{batch_idx}_row_{sample_idx}", f"{prompt_name}.png"))
                    if prompt_name != 'pornstar':
                        grid_data.append((sample['text'][0], gen_images))
                    torch.cuda.empty_cache()

                img_grid_file = os.path.join(full_output_dir, f"grid_{batch_idx}.jpg")
                save_images_grid(grid_data, img_grid_file)




if __name__ == "__main__":
    main()
