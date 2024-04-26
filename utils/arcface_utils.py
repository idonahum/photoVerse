import torch.nn.functional as F
import gdown
import zipfile
import os
from huggingface_hub import hf_hub_download


def setup_arcface_model(root_dir):
    """
    Sets up the ArcFace model and face detection models.
    If the target directory doesn't exist, it creates it and downloads necessary files.
    
    :param root_dir: Root directory for ArcFace model setup.
    """
    # Path for the ArcFace model
    model_path = os.path.join(root_dir, 'models')
    arcface_model_path = os.path.join(model_path, 'antelopev2')
    
    # Check if the target directory exists
    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path, exist_ok=True)

        # Download Face Detection Model
        download_face_detection_models(model_path)
        
        # Load Insightface Pipe (ArcFace)
        hf_hub_download(repo_id="FoivosPar/Arc2Face", filename="arcface.onnx", local_dir=arcface_model_path)
    
    print(f"ArcFace model setup complete in {arcface_model_path}")


def crop_face_from_image(image, face_analysis):
    """
    Given an image and a face analysis object, return a cropped image containing the face.

    Parameters:
    - image: The original image from which to crop.
    - face_analysis: A dictionary containing face analysis data, including 'bbox'.

    Returns:
    - cropped_image: The cropped image containing the face.
    """
    # Extract the bounding box coordinates
    bbox = face_analysis['bbox']
    x_min, y_min, x_max, y_max = bbox.astype(int)  # Convert to integer indices

    # Ensure the indices are within the image boundaries
    h, w = image.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    # Crop the image based on the bounding box
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


def get_largest_bbox_face_analysis(face_analyses):
    """
    Given a list of face analysis objects, return the object with the largest bounding box.

    Parameters:
    - face_analyses: A list of dictionaries containing face analysis data, each with a 'bbox' key.

    Returns:
    - largest_face_analysis: The face analysis object with the largest bounding box.
    """
    if not face_analyses:
        raise ValueError("The face_analyses list is empty.")

    # Sort face analyses by the area of their bounding boxes
    sorted_face_analyses = sorted(
        face_analyses,
        key=lambda fa: (fa['bbox'][2] - fa['bbox'][0]) * (fa['bbox'][3] - fa['bbox'][1]),
        reverse=True  # Largest to smallest
    )

    # Get the face analysis object with the largest bounding box
    largest_face_analysis = sorted_face_analyses[0]

    return largest_face_analysis

def cosine_similarity_between_images(image1, image2, face_analysis_func):
    """
    Given two images, return the cosine similarity of their embeddings.

    Parameters:
    - image1: The first image.
    - image2: The second image.
    - feature_extractor: The feature extractor used to generate embeddings.

    Returns:
    - cosine_similarity_value: The cosine similarity between the embeddings of the two images.
    """
    face_analysis1 = face_analysis_func.get(image1) 
    face_analysis2 = face_analysis_func.get(image2) 

    embedding1 = get_largest_bbox_face_analysis(face_analysis1)['embedding']
    embedding2 = get_largest_bbox_face_analysis(face_analysis2)['embedding']

    # Calculate cosine similarity between the two embeddings
    cosine_similarity_value = F.cosine_similarity(embedding1, embedding2, dim=0).item()  # Get cosine similarity value

    return cosine_similarity_value


def download_face_detection_models(dest_dir, file_id="18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8"):
    """
    Downloads a zip file from Google Drive using the file ID,
    and extracts it to the destination directory if the directory doesn't exist.
    
    :param file_id: Google Drive file ID
    :param dest_dir: Destination directory for extraction
    """
    # Create the direct download URL
    file_url = f"https://drive.google.com/uc?id={file_id}&export=download"

    # Path for the downloaded zip file
    zip_file_path = os.path.join(dest_dir, "downloaded_content.zip")

    # Check if the target directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)  # Create the directory if it doesn't exist

    # Download the zip file
    gdown.download(file_url, zip_file_path, quiet=False)  # Download the zip file

    # Unzip the content to the destination directory
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)  # Extract all contents

    print(f"File downloaded and extracted to {dest_dir}")
