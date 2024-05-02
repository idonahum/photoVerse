import math
import os
import argparse
from pathlib import Path
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from tqdm import tqdm
from diffusers.optimization import get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from huggingface_hub import Repository
import wandb

from datasets.custom import CustomDataset, CustomDatasetWithMasks, collate_fn
from models.infer import run_inference
from models.unet import get_visual_cross_attention_values_norm, set_cross_attention_layers_to_train
from models.modeling_utils import load_models, save_progress
from utils.hub import get_full_repo_name
from utils.image_utils import denormalize, denormalize_clip, to_pil
from insightface.app import FaceAnalysis
from utils.arcface_utils import cosine_similarity_between_images, setup_arcface_model
from PIL import Image

logger = get_logger(__name__)


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
        "--mask_subfolder",
        type=str,
        default=None,
        help="Subfolder relative to data_root_path containing masks",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
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
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for datasets loading. 0 means that the datasets will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    # add num_tokens
    parser.add_argument(
        "--extra_num_tokens",
        type=int,
        default=4,
        help="Number of image encoder hidden states to use as extra tokens for the text encoder",
    )

    parser.add_argument(
        "--image_encoder_layers_idx",
        type=list,
        default=[4, 8, 12, 16],
        help="Image encoder extra layers indices to use as tokens for the text encoder, should be equal to extra_num_tokens",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
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
        default=1.0,
        help='Guidance scale for inference'
    )

    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--arcface_model_root_dir",
        type=str,
        default='arcface',
        help="Destination path for ArcFace models, which include face-detection and embedding models.",
    )

    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def check_args(args):
    if args.extra_num_tokens < 0:
        raise ValueError("extra_num_tokens should be greater than or equal to 0")

    if len(args.image_encoder_layers_idx) != args.extra_num_tokens:
        raise ValueError("The number of image encoder layers to use as tokens should be equal to extra_num_tokens")

    if 0 in args.image_encoder_layers_idx:
        raise ValueError("The image encoder extra tokens layers cant be the last layer since we always use the last layer")

    args.image_encoder_layers_idx = torch.tensor(args.image_encoder_layers_idx)


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        cpu=args.cpu,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    check_args(args)

    face_analysis_func = None
    if args.arcface_model_root_dir is not None:
        setup_arcface_model(args.arcface_model_root_dir)
        face_analysis = FaceAnalysis(name='antelopev2', root=args.arcface_model_root_dir)
        face_analysis.prepare(ctx_id=0, det_size=(args.resolution, args.resolution))
        face_analysis_func = face_analysis.get


    extra_num_tokens = args.extra_num_tokens
    image_encoder_layers_idx = args.image_encoder_layers_idx

    # Load models and tokenizer using the load_models function
    tokenizer, text_encoder, vae, unet, image_encoder, image_adapter, text_adapter, noise_scheduler = load_models(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        extra_num_tokens=extra_num_tokens,
        photoverse_path=args.pretrained_photoverse_path
    )
    
    # optimizer
    # Since we patch unet after freezing, all new parameters are trainable
    unet_params_to_opt = []
    for name, param in unet.named_parameters():
        if param.requires_grad:
            unet_params_to_opt.append(param)

    params_to_opt = itertools.chain(image_adapter.parameters(), text_adapter.parameters(), unet_params_to_opt)
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # dataloader
    if args.mask_subfolder is None:
        train_dataset = CustomDataset(data_root=args.data_root_path, img_subfolder=args.img_subfolder, tokenizer=tokenizer, size=args.resolution)
    else:
        train_dataset = CustomDatasetWithMasks(data_root=args.data_root_path, img_subfolder=args.img_subfolder, tokenizer=tokenizer, size=args.resolution)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        override_max_train_steps = True

    # train
    unet, image_adapter, text_adapter, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet,
        image_adapter,
        text_adapter,
        optimizer,
        train_dataloader,
        lr_scheduler,
        device_placement=[True, True, True, True, False, False])

    # set dtype and device
    weight_dtype = torch.float32
    device = accelerator.device
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    image_encoder.to(device, dtype=weight_dtype)
    image_adapter.to(device, dtype=weight_dtype)
    text_adapter.to(device, dtype=weight_dtype)

    # set to eval
    vae.eval()
    unet.eval()
    image_encoder.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if override_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if accelerator.is_main_process:
        accelerator.init_trackers("photoVerse", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("~~~~~ Running training ~~~~~")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(0, args.num_train_epochs):
        text_adapter.train()
        image_adapter.train()
        set_cross_attention_layers_to_train(unet)
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, image_adapter, text_adapter):
                pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
                pixel_values_clip = batch["pixel_values_clip"].to(device, dtype=weight_dtype)
                placeholder_idx = batch["concept_placeholder_idx"].to(device)
                text_input_ids = batch["text_input_ids"].to(device)

                # Convert images to latent space
                latents = vae.encode(pixel_values).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents).to(latents.device)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # get image_emmbeddings from last layers + extra layers from image_encoder hidden states
                image_features = image_encoder(pixel_values_clip, output_hidden_states=True)
                image_embeddings = [image_features[0]] + [image_features[2][i] for i in image_encoder_layers_idx if i < len(image_features[2])]
                assert len(image_embeddings) == extra_num_tokens + 1, "Entered indices are out of range for image_encoder layers."
                image_embeddings = [emb.detach() for emb in image_embeddings]

                # run through text_adapter
                concept_text_embeddings = text_adapter(image_embeddings)

                encoder_hidden_states = text_encoder({'text_input_ids': text_input_ids,
                                                      "concept_text_embeddings": concept_text_embeddings,
                                                      "concept_placeholder_idx": placeholder_idx.detach()})[0]

                # run through image_adapter
                encoder_hidden_states_image = image_adapter(image_embeddings)

                # Run the UNet
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=(encoder_hidden_states, encoder_hidden_states_image)).sample

                # Calculate concept text regularizer
                concept_text_loss = torch.abs(concept_text_embeddings).mean() * 0.01

                # Calculate reference image regularizer
                cross_attn_values_norm = get_visual_cross_attention_values_norm(unet)
                cross_attn_visual_loss = cross_attn_values_norm.mean() * 0.001

                # Calculate loss
                diffusion_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                loss = diffusion_loss + concept_text_loss + cross_attn_visual_loss

                # Backward
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(text_adapter.parameters(), 1)
                    accelerator.clip_grad_norm_(image_adapter.parameters(), 1)
                    accelerator.clip_grad_norm_(unet.parameters(), 1)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients or True:
                progress_bar.update(1)
                global_step += 1

                input_images = [to_pil(denormalize(img)) for img in batch["pixel_values"]]    
                gen_images = run_inference(batch, tokenizer, image_encoder, text_encoder, unet, text_adapter, image_adapter, vae,
                                            noise_scheduler, device, image_encoder_layers_idx, guidance_scale=args.guidance_scale,
                                            timesteps=args.denoise_timesteps, token_index=0)
            
                if global_step % args.save_steps == 0:
                    save_progress(image_adapter, text_adapter, unet, accelerator, args.output_dir, step=global_step)
                    clip_images = [to_pil(denormalize_clip(img)).resize((train_dataset.size,train_dataset.size)) for img in batch["pixel_values_clip"]]
                    img_list = []
                    for gen_img, input_img, clip_img in zip(gen_images, input_images, clip_images):
                        img_list.append(np.concatenate((np.array(gen_img), np.array(input_img), np.array(clip_img)), axis=1))
                    img_list = np.concatenate(img_list, axis=0)
                    img_grid = Image.fromarray(img_list)
                    img_grid_file = os.path.join(args.output_dir, f"{str(global_step).zfill(5)}.jpg")
                    img_grid.save(img_grid_file)
                    if args.report_to == "wandb":
                        images = wandb.Image(img_grid_file, caption="Generated images vs input images")
                        logs = {"Generated images vs input images": images}
                        accelerator.log(logs, step=global_step)

            if face_analysis_func is not None:
                # TODO: its slow, not efficient (need to use Torch, ONNX has some issue with GPU), and should be edited to a more stable loss.
                # MAYBE: https://medium.com/@payyavulasaiprakash/arcface-loss-function-for-deep-face-recognition-e1ff5e173b52
                arcface_score = np.mean([cosine_similarity_between_images(input_image, gen_image, face_analysis_func) for input_image, gen_image in zip(input_images, gen_images)])
                accelerator.log({"arcface_loss": 1 - abs(arcface_score)})

            logs = {"loss_mle": diffusion_loss.detach().item(),
                    "loss_reg_concept_text": concept_text_loss.detach().item(),
                    "loss_reg_cross_attn_visual": cross_attn_visual_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_progress(image_adapter, text_adapter, unet, accelerator, args)

    accelerator.end_training()


if __name__ == "__main__":
    main()
