import numpy as np
import cv2
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from models.arcface_resnet import ArcFaceResNet18
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class FaceSimilarity:
    def __init__(self, model_name='arcface', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.mtcnn = MTCNN(device=device).eval()
        self.model = self._load_model(model_name).eval().to(device)
        self.input_size = 128 if model_name == 'arcface' else 160
        self.device = device
        self.model_name = model_name

    @staticmethod
    def _load_model(model_name):
        if model_name == 'arcface':
            return ArcFaceResNet18(pretrained=True)
        else:
            return InceptionResnetV1(pretrained='vggface2')

    def preprocess_image(self,image):
        """
        Convert image from PIL to numpy array and convert from BGR to RGB.
        """
        image_np = np.array(image)
        image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        if self.model_name == 'arcface':
            image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        return image

    def extract_features(self, image, box):
        """
        Extract features for detected faces in the image.
        """
        x1, y1, x2, y2 = box.astype(int)
        if image[y1:y2, x1:x2].size == 0:
            return None
        
        # Proceed with processing the face image
        face = cv2.resize(image[y1:y2, x1:x2], (160, 160))
        face = (face / 255.0 - 0.5) / 0.5
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        with torch.no_grad():
            embedding = self.model(torch.tensor(face).float().to(self.device)).squeeze().numpy()
        
        return embedding

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Compute cosine similarity between two vectors.
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def __call__(self, image1, image2):
        return self.calculate_face_similarity(image1, image2)

    def calculate_face_similarity(self, image1, image2):
        """
        Calculate cosine similarity between faces in two images.
        """
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)
        boxes1, _ = self.mtcnn.detect(image1)
        boxes2, _ = self.mtcnn.detect(image2)

        # If either image is not recognized, return similarity score of 0
        if boxes1 is None or boxes2 is None \
            or len(boxes1) == 0 or len(boxes2) == 0:
            return 0.0

        # Select the largest bounding box for each image based on area
        best_box1 = self.select_largest_bbox(boxes1)
        best_box2 = self.select_largest_bbox(boxes2)

        embedding1 = self.extract_features(image1, best_box1)
        embedding2 = self.extract_features(image2, best_box2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        similarity = self.cosine_similarity(embedding1, embedding2)
        return similarity

    @staticmethod
    def select_largest_bbox(boxes):
        """
        Select the largest bounding box based on area.
        """
        # Sort boxes based on area in descending order
        sorted_boxes = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
        
        # Return the first (largest) bounding box
        return sorted_boxes[0]


if __name__ == "__main__":
    facenet_similarity = FaceSimilarity()

    image_pairs = [
        ('results/2.png', 'results/2.png'),
        ('results/8.jpg', 'results/8.png'),
        ('results/15.jpg', 'results/15.png'),
        ('results/25.jpg', 'results/25.png'),
        ('results/100.jpg', 'results/100.png')
        # Add more pairs of images here
    ]

    # Load images using PIL and calculate similarity for each pair of images
    for image_pair in image_pairs:
        with Image.open(image_pair[0]) as image1, Image.open(image_pair[1]) as image2:
            similarity = facenet_similarity.calculate_face_similarity(image1, image2)
            print(f"Cosine Similarity between faces in {image_pair[0]} and {image_pair[1]}:", similarity)
