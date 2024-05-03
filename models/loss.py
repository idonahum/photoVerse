from utils.arcface_utils import cosine_similarity_between_images
import torch


def face_loss(input_images, gen_images, face_analysis_func):
    arcface_scores = torch.tensor([cosine_similarity_between_images(gen_image, input_image, face_analysis_func) for gen_image, input_image in zip(gen_images, input_images)], dtype=torch.float32)
    loss = 1.0 - torch.abs(arcface_scores.mean())
    return loss