from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import numpy as np


def denormalize(image):
    """
    Denormalize the image from [-1, 1] to [0, 1]
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


def save_images_grid(grid_data, img_grid_file):
    img_list = []
    text_data = [text_image_pair[0] for text_image_pair in grid_data]
    image_data = [text_image_pair[1] for text_image_pair in grid_data]
    for zipped_images in zip(*image_data):
        # Concatenate images horizontally
        img_list.append(
            np.concatenate([np.array(image) for image in zipped_images], axis=1))

    # Concatenate all rows vertically
    img_array = np.concatenate(img_list, axis=0)
    padded_img_array = np.pad(img_array, ((50, 0), (0, 0), (0, 0)), mode='constant', constant_values=255)
    img_grid = Image.fromarray(padded_img_array.astype('uint8'), 'RGB')

    # Create a drawing context
    draw = ImageDraw.Draw(img_grid)

    # You may need to install a font or specify a path to one that supports the size you need
    # For basic purposes, you can use a default PIL font:
    try:
        font = ImageFont.truetype("arial.ttf", 36)  # Specify the font and size you need
    except IOError:
        font = ImageFont.load_default(size=36)

    num_of_images_in_row = len(image_data)
    image_width = image_data[0][0].width
    for i, text in enumerate(text_data):
        # Calculate the position for the text to be at the center above the i row
        text = text.format("S*")
        text_x1, text_y1, text_x2, text_y2 = font.getbbox(text)
        text_width = text_x2 - text_x1
        text_height = text_y2 - text_y1
        text_x = (image_width - text_width) // 2 + i * image_width
        text_y = (50 - text_height) // 2

        # Draw the text
        draw.text((text_x, text_y), text, font=font, fill="black")

    img_grid.save(img_grid_file)


if __name__ == '__main__':
    import os
    # Define the prompts and their corresponding filenames
    prompts = [
        "Input Image",
        "A photo of S*",
        "S* in Ghibli anime style",
        "S* wears a red hat",
        "S* on the beach",
        "Manga drawing of S*",
        "S* as a Funko Pop figure",
        "S* stained glass window",
        "Watercolor painting of S*"
    ]

    # Define the base directory where the images are located
    base_dir = "../figs"
    grid_data = []

    for prompt in prompts:
        images = []
        for i in range(1, 6):
            # Construct the filename based on the prompt and index
            file_name_map = {
                "Input Image": f"input_image{i}.png",
                "A photo of S*": f"photo{i}.png",
                "S* in Ghibli anime style": f"ghibli{i}.png",
                "S* wears a red hat": f"red_hat{i}.png",
                "S* on the beach": f"beach{i}.png",
                "Manga drawing of S*": f"manga{i}.png",
                "S* as a Funko Pop figure": f"funko_pop{i}.png",
                "S* stained glass window": f"stained_glass{i}.png",
                "Watercolor painting of S*": f"watercolor{i}.png"
            }
            image_path = os.path.join(base_dir, str(i), file_name_map[prompt])
            images.append(Image.open(image_path))
        grid_data.append((prompt, images))

    # Save the image grid
    save_images_grid(grid_data, "image_grid.png")