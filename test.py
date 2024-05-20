import argparse
import os
import torch
import logging
from insightface.app import FaceAnalysis

from datasets.custom import CustomDataset, collate_fn
from models.infer import run_inference
from models.modeling_utils import load_models
from utils.arcface_utils import setup_arcface_model, cosine_similarity_between_images
from utils.image_utils import to_pil, denormalize, save_images_grid


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
        "--arcface_model_root_dir",
        type=str,
        default="arcface_model",
        help="The root directory for the arcface model",
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
    # Configure logging
    model_name = args.pretrained_photoverse_path.split('/')[-1]
    model_name = model_name.split('.')[0]
    logging.basicConfig(filename=os.path.join(args.output_dir, f'{model_name}_tsp_{args.denoise_timesteps}.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info(args)
    image_encoder_layers_idx = torch.tensor(args.image_encoder_layers_idx).to(args.device)

    setup_arcface_model(args.arcface_model_root_dir)
    face_analysis = FaceAnalysis(name='antelopev2', root=args.arcface_model_root_dir)
    face_analysis.prepare(ctx_id=0, det_size=(args.resolution, args.resolution))
    face_analysis_func = face_analysis.get
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
        similarity_list = []

        full_output_dir = os.path.join(args.output_dir, f"{model_name}_{args.denoise_timesteps}_timesteps", split)
        os.makedirs(full_output_dir, exist_ok=True)
        for batch_idx, sample in enumerate(dataloader):
            if (batch_idx + 1)*args.batch_size > args.max_gen_images:
                print((batch_idx + 1)*args.batch_size)
                break
            with torch.no_grad():
                print(f"Batch: {batch_idx}")
                pixel_values = sample["pixel_values"].to(args.device)
                gen_tensors = run_inference(sample, tokenizer, image_encoder, text_encoder, unet, text_adapter,
                                            image_adapter, vae,
                                            scheduler, args.device, image_encoder_layers_idx,
                                            guidance_scale=args.guidance_scale,
                                            timesteps=args.denoise_timesteps, token_index=0)
                gen_images = [to_pil(denormalize(gen_tensor)) for gen_tensor in gen_tensors]
                input_images = [to_pil(denormalize(img)) for img in pixel_values]
                grid_data = [("Input Images", input_images), ("Generated Images", gen_images)]
                img_grid_file = os.path.join(full_output_dir, f"grid_{batch_idx}.jpg")
                save_images_grid(grid_data, img_grid_file)

                for sample_idx, (gen_image, input_image) in enumerate(zip(gen_images, input_images)):
                    similarity_list.append(cosine_similarity_between_images(gen_image, input_image, face_analysis_func))
                    gen_image.save(os.path.join(full_output_dir, f"generated_img_batch_idx{batch_idx}_sample_idx{sample_idx}.png"))
                    input_image.save(os.path.join(full_output_dir, f"input_img_batch_idx{batch_idx}_sample_idx{sample_idx}.png"))

                torch.cuda.empty_cache()
                logging.info(f"Current Average similarity: {sum(similarity_list) / len(similarity_list)}, Split type: {split}, Model: {model_name}, Timesteps: {args.denoise_timesteps}")

        logging.info(f"Final similarity: {sum(similarity_list)/len(similarity_list)}, Split type: {split}, Model: {model_name}, Timesteps: {args.denoise_timesteps}")


if __name__ == "__main__":
    main()
