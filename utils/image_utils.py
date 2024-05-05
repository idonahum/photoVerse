from PIL import Image
from torchvision import transforms
import numpy as np

def denormalize(image):
    """
    Denormalize the image from [-1, 1] to [0, 255]
    """
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def denormalize_clip(image, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    """
    Denormalize the image based on mean and std which used for normalization, default values are for CLIP
    """
    # Denormalize the image with pytorch Normalize
    image = transforms.Normalize([-m/s for m, s in zip(mean, std)], [1/s for s in std])(image)
    return image


def to_pil(image):
    """
    Convert the image from tensor to PIL image
    """
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image)


def save_images_grid(gen_images, input_images, clip_images, img_grid_file):
    img_list = []
    for gen_img, input_img, clip_img in zip(gen_images, input_images, clip_images):
        img_list.append(
            np.concatenate((np.array(gen_img), np.array(input_img), np.array(clip_img)), axis=1))
    img_list = np.concatenate(img_list, axis=0)
    img_grid = Image.fromarray(img_list)

    img_grid.save(img_grid_file)