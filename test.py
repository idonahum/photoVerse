import argparse
import os
import torch

from datasets.custom import CustomDataset, collate_fn
from models.infer import run_inference
from models.loss import FaceLoss
from models.modeling_utils import load_models
from utils.image_utils import to_pil, denormalize


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
    return parser.parse_args()


def main():
    args = parse_args()
    face_similarity = FaceLoss(device=args.device, model_name='arcface')
    tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter, scheduler, _ = load_models(
        args.pretrained_model_name_or_path, args.extra_num_tokens, args.pretrained_photoverse_path)

    test_dataset = CustomDataset(data_root=args.data_root_path, img_subfolder=args.img_subfolder,
                  tokenizer=tokenizer, size=args.resolution)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    image_encoder_layers_idx = torch.tensor(args.image_encoder_layers_idx).to(args.device)
    similarity_list = []
    idx = 0
    model_name = args.pretrained_model_name_or_path.split('/')[-1]
    # remove the file extension
    model_name = model_name.split('.')[0]
    full_output_dir = os.path.join(args.output_dir, f"{model_name}_{args.denoise_timesteps}_timesteps")
    os.makedirs(full_output_dir, exist_ok=True)
    for sample in test_dataloader:
        with torch.no_grad():
            pixel_values = sample["pixel_values"].to(args.device)
            gen_tensors = run_inference(sample, tokenizer, image_encoder, text_encoder, unet, text_adapter,
                                        image_adapter, vae,
                                        scheduler, args.device, image_encoder_layers_idx,
                                        guidance_scale=args.guidance_scale,
                                        timesteps=args.denoise_timesteps, token_index=0)
            similarity_metric = face_similarity(pixel_values, gen_tensors, normalize=False,
                                          maximize=False).detach().item()
            similarity_list.append(similarity_metric)
            generated_images = [to_pil(denormalize(img)) for img in generated_images]

            for img in generated_images:
                img.save(os.path.join(full_output_dir, f"image_{idx}.png"))
                idx += 1
    print(f"Num of timesteps: {args.denoise_timesteps}")
    print(f"Average similarity: {sum(similarity_list)/len(similarity_list)}")

if __name__ == "__main__":
    main()